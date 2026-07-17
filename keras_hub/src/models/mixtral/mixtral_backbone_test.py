import gc
import json
import re

import keras
import pytest
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.mixtral.mixtral_backbone import MixtralBackbone
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.preset_utils import CONFIG_FILE
from keras_hub.src.utils.preset_utils import get_file

# Dims for the Tier-2 CI-safe mesh-shape sweep: representative real-preset
# dimensions, frozen as literals and sourced once, offline -- do not add a
# `get_file` call to the Tier-2 test body itself (that's what Tier 3,
# `test_layout_map_live_presets` below, is for).
#
# MEMORY NOTE: this local dev machine cannot load full-scale model dims (see
# gemma_backbone_test.py's identical note for the OOM history that
# established this rule). What actually matters for the divisibility/
# sharding properties this tier tests is the RATIO of query heads to kv
# heads (and the MoE expert/top_k config) -- not the absolute parameter
# count. So these dims are scaled down by roughly 20-30x from the single
# real Mixtral preset (`mixtral_8_7b_en` / `mixtral_8_instruct_7b_en`, which
# share one architecture: vocabulary_size=32000, hidden_dim=4096,
# intermediate_dim=14336, num_query_heads=32, num_key_value_heads=8 (4:1
# GQA), head_dim=128, num_experts=8, top_k=2 -- confirmed via a live
# `get_file(preset, CONFIG_FILE)` config fetch, 2026-07-17) while preserving
# that exact 4:1 query:kv head ratio and the 8-expert/top-2 MoE shape, and
# keeping hidden/intermediate/vocab divisible by every mesh shape in
# CAPPED_MESH_SHAPES. Two width classes (small/base) are derived at the same
# ratio and MoE shape but different absolute scale, since Mixtral (unlike
# Gemma) has only one real preset architecture to source ratios from -- this
# still exercises the divisibility sweep at more than one scale. Full-scale
# real dims are exercised by `test_layout_map_live_presets` below, which has
# its own per-width-class memory-budget skip so it never attempts a
# full-scale build locally either -- true full-scale verification happens
# offline on a machine with more RAM (Tier 4 in the design doc), not on this
# box.
MIXTRAL_SMALL_DIMS = {
    # Single-token label (no space before the parenthetical) so
    # `.split(' ')[0]` below yields a name distinct from
    # MIXTRAL_BASE_DIMS's -- both are scaled from the same real preset, so
    # a naive "mixtral_8_7b_en" prefix on both collided into duplicate
    # parameterized test names (absl.testing.DuplicateTestNameError).
    "source_preset": ("mixtral8x7b-small (real ratio, memory-scaled dims, x1)"),
    "vocabulary_size": 1600,
    "num_layers": 1,  # depth is irrelevant to spec matching/divisibility.
    "num_query_heads": 8,
    "num_key_value_heads": 2,  # real ratio: GQA, 4:1.
    "hidden_dim": 128,
    "intermediate_dim": 512,
    "num_experts": 8,  # real preset value, unscaled -- already tiny.
    "top_k": 2,
    "sliding_window": None,
}
MIXTRAL_BASE_DIMS = {
    "source_preset": ("mixtral8x7b-base (real ratio, memory-scaled dims, x2)"),
    "vocabulary_size": 3200,
    "num_layers": 1,
    "num_query_heads": 16,
    "num_key_value_heads": 4,  # real ratio: GQA, 4:1.
    "hidden_dim": 256,
    "intermediate_dim": 1024,
    "num_experts": 8,
    "top_k": 2,
    "sliding_window": None,
}

# Hard-capped mesh-shape list for this shared 37GB dev machine. The full
# 10-shape matrix from the testing-strategy doc is
# 2x4, 1x8, 4x4, 8x8, 16x16, 2x2x2, 1x1x8, 2x2x4, 4x4x4, 4x4x8 -- shapes
# 8x8, 16x16, 4x4x4, 4x4x8 (64-256 virtual devices) are DELIBERATELY
# DROPPED here due to a demonstrated systemd-oomd OOM kill of the entire
# desktop app on this shared machine during an earlier attempt at this same
# pipeline (see gemma_backbone_test.py's identical comment). Do not attempt
# the dropped shapes even experimentally on this box -- revisiting them
# requires a dedicated or CI machine, not this one.
CAPPED_MESH_SHAPES = [
    (2, 4),
    (1, 8),
    (4, 4),
    (2, 2, 2),
    (1, 1, 8),
    (2, 2, 4),
]

# Same expected_shardings patterns as MixtralBackboneTest.test_distribution
# (post-QKV-axis-fix), reused by the Tier-2 and Tier-3 mesh sweeps below.
# MoE expert-bank and router rules were already correct pre-fix and are
# unchanged here -- included for full coverage-assertion completeness.
_EXPECTED_SHARDINGS = {
    "token_embedding/embeddings": ("model", "batch"),
    "token_embedding/reverse_embeddings": ("batch", "model"),
    "self_attention/query/kernel": ("batch", "model", None),
    "self_attention/key/kernel": ("batch", None, None),
    "self_attention/value/kernel": ("batch", None, None),
    "self_attention/attention_output/kernel": ("model", None, "batch"),
    "experts/expert_feedforward_gate_dense": (None, "batch", "model"),
    "experts/expert_feedforward_intermediate_dense": (
        None,
        "batch",
        "model",
    ),
    "experts/expert_feedforward_output_dense": (None, "model", "batch"),
    "sparse_feedforward_gate_dense/kernel": ("batch", None),
}


def _assert_mixtral_shardings_and_coverage(test_case, model, layout_map):
    """Shared spec + coverage assertions for the Tier-2/3 mesh sweeps."""
    for pattern, expected in _EXPECTED_SHARDINGS.items():
        matches = [w for w in model.weights if re.search(pattern, w.path)]
        test_case.assertGreater(len(matches), 0)
        for w in matches:
            test_case.assertEqual(tuple(w.value.sharding.spec), expected)
    offending = [
        w.path
        for w in model.weights
        if len(w.shape) >= 2 and layout_map[w.path] is None
    ]
    test_case.assertEqual(
        offending,
        [],
        f"The following rank>=2 weights are unmapped: {offending}",
    )


class MixtralBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_query_heads": 8,
            "num_key_value_heads": 4,
            "hidden_dim": 16,
            "intermediate_dim": 8,
            "num_experts": 2,
            "top_k": 2,
            "sliding_window": 2,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=MixtralBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 16),
            run_quantization_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MixtralBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_num_parameters(self):
        model = MixtralBackbone(**self.init_kwargs)
        # Calculated based on the model architecture:
        # - Token embedding: vocabulary_size * hidden_dim + hidden_dim *
        # vocabulary_size (tie_weights=False)
        # - Transformer layers: 2 * (attention + MoE block + layer norms)
        # - Attention: query + key + value + output
        # - MoE: experts (gate + intermediate + output) + router
        # - Layer norms: hidden_dim each
        head_dim = 16 // 8  # hidden_dim / num_query_heads
        expected_params = (
            10 * 16
            + 16 * 10  # Token embedding (embedding + output projection)
            + 2
            * (  # Two layers
                (  # Attention
                    16 * head_dim * 8  # Query
                    + 16 * head_dim * 4  # Key
                    + 16 * head_dim * 4  # Value
                    + 8 * head_dim * 16  # Output
                )
                + (  # MoE
                    2 * (16 * 8 + 16 * 8 + 8 * 16) + 16 * 2
                )
                + 2 * 16  # Two layer norms (self_attention + feedforward)
            )
            + 16  # Final layer norm
        )
        self.assertEqual(model.count_params(), expected_params)

    def test_distribution(self):
        self.run_distribution_test(
            cls=MixtralBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_shardings={
                "token_embedding/embeddings": ("model", "batch"),
                "token_embedding/reverse_embeddings": ("batch", "model"),
                "self_attention/query/kernel": ("batch", "model", None),
                "self_attention/key/kernel": ("batch", None, None),
                "self_attention/value/kernel": ("batch", None, None),
                "self_attention/attention_output/kernel": (
                    "model",
                    None,
                    "batch",
                ),
                "experts/expert_feedforward_gate_dense": (
                    None,
                    "batch",
                    "model",
                ),
                "experts/expert_feedforward_intermediate_dense": (
                    None,
                    "batch",
                    "model",
                ),
                "experts/expert_feedforward_output_dense": (
                    None,
                    "model",
                    "batch",
                ),
                "sparse_feedforward_gate_dense/kernel": ("batch", None),
            },
            allow_replicated=(),
            # MoE: sharded-reduction float noise is larger than dense
            # models -- see the helper docstring and the plan's Section 0.1
            # addendum (verified empirically on Mixtral itself:
            # max_abs_diff=2.18e-6 at mesh (2,2), diagnosed as ordinary
            # float noise, not a routing flip).
            is_moe=True,
        )

    @parameterized.named_parameters(
        (
            f"{dims['source_preset'].split(' ')[0]}_mesh"
            f"_{'x'.join(str(s) for s in shape)}",
            dims,
            shape,
        )
        for dims in (MIXTRAL_SMALL_DIMS, MIXTRAL_BASE_DIMS)
        for shape in CAPPED_MESH_SHAPES
    )
    def test_layout_map_mesh_shapes(self, dims, mesh_shape):
        if keras.backend.backend() != "jax":
            self.skipTest("`ModelParallel` testing requires the Jax backend.")
        devices = keras.distribution.list_devices("CPU")
        n_needed = 1
        for s in mesh_shape:
            n_needed *= s
        if n_needed > len(devices):
            self.skipTest(
                f"Mesh shape {mesh_shape} needs {n_needed} devices, only "
                f"{len(devices)} available. Run with "
                f"XLA_FLAGS=--xla_force_host_platform_device_count="
                f"{n_needed} to exercise this shape locally."
            )

        # Query-head-divisibility skip rule: an inherent tensor-parallelism
        # ceiling, not a bug. model_axis_size is the mesh's last axis,
        # matching this repo's axis_names=(..., "model") convention. Key/
        # value heads are never sharded on the model axis under the
        # corrected map (always replicated, see get_layout_map's comment),
        # so there is no analogous kv-divisibility skip needed here.
        model_axis_size = mesh_shape[-1]
        num_query_heads = dims["num_query_heads"]
        if num_query_heads % model_axis_size != 0:
            self.skipTest(
                f"num_query_heads={num_query_heads} not divisible by "
                f"model-axis={model_axis_size}: inherent "
                "tensor-parallelism limit, not a bug"
            )

        devices = devices[:n_needed]
        if len(mesh_shape) == 2:
            axis_names = ("batch", "model")
        else:
            # 3D shape: the extra axis is named "seq" per the plan/
            # testing-strategy convention, but get_layout_map only names
            # "batch"/"model" -- the "seq" axis simply replicates weights
            # (no rule targets it), matching every other 3D-mesh test in
            # this PR series.
            axis_names = ("batch", "seq", "model")
        device_mesh = keras.distribution.DeviceMesh(
            shape=mesh_shape,
            axis_names=axis_names,
            devices=devices,
        )
        layout_map = MixtralBackbone.get_layout_map(device_mesh)
        distribution = keras.distribution.ModelParallel(
            layout_map=layout_map, batch_dim_name="batch"
        )
        init_kwargs = {k: v for k, v in dims.items() if k != "source_preset"}
        with distribution.scope():
            # bfloat16: a memory mitigation for this shared dev machine --
            # spec assertions are dtype-independent.
            model = MixtralBackbone(dtype="bfloat16", **init_kwargs)
            _assert_mixtral_shardings_and_coverage(self, model, layout_map)
        del model
        gc.collect()

    @pytest.mark.kaggle_key_required
    @pytest.mark.multi_device
    @pytest.mark.extra_large
    def test_layout_map_live_presets(self):
        if keras.backend.backend() != "jax":
            self.skipTest("`ModelParallel` testing requires the Jax backend.")

        # Fetch every preset's config only (no weights), then dedupe by the
        # divisibility-relevant dims so width-classes that share a config
        # (mixtral_8_7b_en and mixtral_8_instruct_7b_en share one identical
        # architecture, confirmed via a live config fetch 2026-07-17) are
        # only built once per mesh shape -- a memory/time necessity on this
        # machine (mixtral_8_7b-width builds are ~1.8GB estimated even at 1
        # layer bf16, see the memory-budget guard below), while every preset
        # in the registry is still fetched and evaluated, preserving full
        # registry coverage.
        dim_keys = (
            "vocabulary_size",
            "num_query_heads",
            "num_key_value_heads",
            "hidden_dim",
            "intermediate_dim",
            "num_experts",
            "top_k",
        )
        width_classes = {}  # dedupe key -> (config dict, [preset names])
        fetch_failures = []
        for preset in MixtralBackbone.presets:
            try:
                path = get_file(preset, CONFIG_FILE)
                cfg = json.load(open(path))["config"]
            except Exception as e:
                # A preset this account can't reach (e.g. an unaccepted
                # Kaggle license consent click-through) is logged, not
                # fatal -- the rest of the registry still gets exercised.
                fetch_failures.append((preset, str(e)))
                continue
            # num_layers is forced to 1 below regardless of the real
            # value -- layout rules are per-decoder-block regexes, so
            # depth is irrelevant to spec matching/divisibility, and 1
            # layer keeps build memory bounded on this shared machine.
            cfg = dict(cfg)
            cfg["num_layers"] = 1
            key = tuple(cfg.get(k) for k in dim_keys)
            if key not in width_classes:
                width_classes[key] = (cfg, [])
            width_classes[key][1].append(preset)

        if fetch_failures:
            print(
                f"test_layout_map_live_presets: {len(fetch_failures)} "
                f"preset config fetches failed (logged, non-fatal): "
                f"{fetch_failures}"
            )
        if not width_classes:
            self.skipTest(
                "No preset configs were reachable "
                f"({len(fetch_failures)} fetch failures) -- likely a "
                "Kaggle license-consent gate on this account for the "
                "mixtral family. See the module comment above "
                "CAPPED_MESH_SHAPES."
            )
        print(
            f"test_layout_map_live_presets: {len(width_classes)} unique "
            f"width-classes across {len(MixtralBackbone.presets)} registry "
            "presets:"
        )
        for cfg, presets in width_classes.values():
            print(f"  {presets}")

        devices = keras.distribution.list_devices("CPU")
        skip_reasons = []
        ran_any = False
        for cfg, presets in width_classes.values():
            num_query_heads = cfg["num_query_heads"]
            for mesh_shape in CAPPED_MESH_SHAPES:
                combo_label = f"{presets[0]}@{mesh_shape}"
                with self.subTest(combo=combo_label):
                    n_needed = 1
                    for s in mesh_shape:
                        n_needed *= s
                    if n_needed > len(devices):
                        reason = (
                            f"{combo_label}: needs {n_needed} devices, "
                            f"only {len(devices)} available"
                        )
                        skip_reasons.append(reason)
                        continue
                    model_axis_size = mesh_shape[-1]
                    if num_query_heads % model_axis_size != 0:
                        reason = (
                            f"{combo_label}: num_query_heads="
                            f"{num_query_heads} not divisible by "
                            f"model-axis={model_axis_size}: inherent "
                            "tensor-parallelism limit, not a bug"
                        )
                        skip_reasons.append(reason)
                        continue

                    # Memory-budget guard: this shared dev machine cannot
                    # locally build full-scale presets (see
                    # gemma_backbone_test.py's identical guard and its OOM
                    # history). Estimate this width-class's single-decoder-
                    # block bf16 footprint (embedding table + one FFN
                    # block's 3 matrices, times a 3x safety margin for
                    # JAX/XLA transient copies during construction/
                    # resharding) and skip the actual build if it exceeds a
                    # conservative local threshold. The config-fetch,
                    # dedup, and divisibility-skip logic above still
                    # exercises every registry preset either way; only the
                    # expensive build+assert step is capped. Note this
                    # formula intentionally does not add the MoE expert
                    # banks (num_experts x 3 matrices) -- it mirrors the
                    # dense-model estimate other models in this series use,
                    # which already produces a comfortably-over-threshold
                    # estimate for Mixtral's real preset (~1.8GB, verified
                    # 2026-07-17), so the omission does not change the
                    # skip outcome; it is not claimed to be a tight bound.
                    est_params = (
                        cfg["vocabulary_size"] * cfg["hidden_dim"]
                        + 3 * cfg["hidden_dim"] * cfg["intermediate_dim"]
                    )
                    est_bytes = est_params * 2 * 3  # bf16 * safety margin
                    max_local_bytes = 300 * 1024 * 1024  # 300MB
                    if est_bytes > max_local_bytes:
                        reason = (
                            f"{combo_label}: estimated build memory "
                            f"~{est_bytes / 1e9:.2f}GB exceeds the "
                            f"{max_local_bytes / 1e6:.0f}MB local safety "
                            "threshold on this shared, RAM-constrained "
                            "dev machine -- verify this width-class on a "
                            "machine with more RAM or in CI"
                        )
                        skip_reasons.append(reason)
                        continue

                    combo_devices = devices[:n_needed]
                    if len(mesh_shape) == 2:
                        axis_names = ("batch", "model")
                    else:
                        axis_names = ("batch", "seq", "model")
                    device_mesh = keras.distribution.DeviceMesh(
                        shape=mesh_shape,
                        axis_names=axis_names,
                        devices=combo_devices,
                    )
                    layout_map = MixtralBackbone.get_layout_map(device_mesh)
                    distribution = keras.distribution.ModelParallel(
                        layout_map=layout_map, batch_dim_name="batch"
                    )
                    init_kwargs = {
                        k: v for k, v in cfg.items() if k in dim_keys
                    }
                    init_kwargs["num_layers"] = 1
                    with distribution.scope():
                        model = MixtralBackbone(dtype="bfloat16", **init_kwargs)
                        _assert_mixtral_shardings_and_coverage(
                            self, model, layout_map
                        )
                    del model
                    gc.collect()
                    ran_any = True

        print(
            f"test_layout_map_live_presets: {len(skip_reasons)} combo(s) "
            f"skipped:\n" + "\n".join(f"  {r}" for r in skip_reasons)
        )
        if not ran_any:
            self.skipTest(
                "All (width-class, mesh-shape) combos were skipped: "
                f"{skip_reasons}"
            )
