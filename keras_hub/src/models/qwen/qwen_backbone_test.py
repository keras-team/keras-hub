import gc
import json
import os
import re

import keras
import pytest
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.qwen.qwen_backbone import QwenBackbone
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.preset_utils import CONFIG_FILE
from keras_hub.src.utils.preset_utils import get_file

# Dims for the Tier-2 CI-safe mesh-shape sweep: representative real-preset
# dimensions, frozen as literals and sourced once, offline -- do not add a
# `get_file` call to the Tier-2 test body itself (that's what Tier 3,
# `test_layout_map_live_presets` below, is for).
#
# MEMORY NOTE: full-scale Qwen dims are impractical to build in
# memory-constrained local environments. What actually matters for the
# divisibility/sharding properties this tier tests is the RATIO of query
# heads to key/value heads and whether hidden/intermediate/vocab divide the
# mesh's model-axis sizes -- not the absolute parameter count. So these dims
# are scaled down by roughly 20-30x from the real Qwen2.5 presets (confirmed
# by fetching their live config.json files) while preserving each preset's
# EXACT real query:kv head ratio (14:2, 16:2, 28:4) and keeping head_dim
# modest (32). num_layers is always 1 -- layout rules are per-decoder-block
# regexes, so depth is irrelevant to spec matching/divisibility. The real
# head counts are kept verbatim (not rounded to a power of 2) so the 14-head
# and 28-head classes genuinely exercise the num_query_heads-not-divisible-
# by-8 skip path, while the 16-head class builds at every capped mesh shape.
# Full-scale real dims are exercised by `test_layout_map_live_presets` below,
# which has its own per-width-class memory-budget skip so it never attempts
# a full-scale build locally either -- true full-scale verification happens
# on a machine with more RAM or in CI.
QWEN_0_5B_DIMS = {
    "source_preset": "qwen2.5_0.5b_en (real ratio, memory-scaled dims)",
    "vocabulary_size": 2048,
    "num_layers": 1,  # depth is irrelevant to spec matching/divisibility.
    "num_query_heads": 14,
    "num_key_value_heads": 2,  # real ratio: GQA, 7:1.
    "hidden_dim": 448,  # num_query_heads * head_dim (14 * 32).
    "intermediate_dim": 1024,
    "head_dim": 32,
}
QWEN_3B_DIMS = {
    "source_preset": "qwen2.5_3b_en (real ratio, memory-scaled dims)",
    "vocabulary_size": 2048,
    "num_layers": 1,
    "num_query_heads": 16,
    "num_key_value_heads": 2,  # real ratio: GQA, 8:1.
    "hidden_dim": 512,  # num_query_heads * head_dim (16 * 32).
    "intermediate_dim": 1536,
    "head_dim": 32,
}
QWEN_7B_DIMS = {
    "source_preset": "qwen2.5_7b_en (real ratio, memory-scaled dims)",
    "vocabulary_size": 2048,
    "num_layers": 1,
    "num_query_heads": 28,
    "num_key_value_heads": 4,  # real ratio: GQA, 7:1.
    "hidden_dim": 896,  # num_query_heads * head_dim (28 * 32).
    "intermediate_dim": 2048,
    "head_dim": 32,
}

# Hard-capped mesh-shape list for memory-constrained local environments. The
# full 10-shape matrix from the testing-strategy doc is
# 2x4, 1x8, 4x4, 8x8, 16x16, 2x2x2, 1x1x8, 2x2x4, 4x4x4, 4x4x8 -- shapes
# 8x8, 16x16, 4x4x4, 4x4x8 (64-256 virtual devices) are DELIBERATELY
# DROPPED here due to a demonstrated OOM kill during an earlier attempt at
# this same pipeline on a memory-constrained machine. These shapes require a
# dedicated or CI machine with more memory -- do not attempt them
# experimentally in a constrained local environment.
CAPPED_MESH_SHAPES = [
    (2, 4),
    (1, 8),
    (4, 4),
    (2, 2, 2),
    (1, 1, 8),
    (2, 2, 4),
]

# Post-fix expected shardings, reused by the Tier-2 and Tier-3 mesh sweeps.
# The query kernel is (hidden, num_query_heads, head_dim) -> query heads on
# the model axis (Megatron column-parallel), contracting hidden on data. The
# key/value kernels are (hidden, num_key_value_heads, head_dim) -> kv heads
# replicated on the model axis (small under GQA), hidden on data. These use
# the mesh's own axis names ("model"/"batch"), matching this repo's
# axis_names=(..., "model") convention where model_dim is the last axis.
_EXPECTED_SHARDINGS = {
    "token_embedding/embeddings": ("model", "batch"),
    "self_attention.*query.kernel": ("batch", "model", None),
    "self_attention.*key.kernel": ("batch", None, None),
    "self_attention.*value.kernel": ("batch", None, None),
    "self_attention.*attention_output.kernel": ("model", None, "batch"),
    "feedforward_intermediate_dense.kernel": ("batch", "model"),
    "feedforward_gate_dense.kernel": ("batch", "model"),
    "feedforward_output_dense.kernel": ("model", "batch"),
}

# The q/k/v EinsumDense layers use `bias_axes`, producing rank-2 biases of
# shape (num_heads, head_dim) that no `.kernel` layout rule targets and that
# are intentionally left replicated (negligible size; sharding them would add
# divisibility risk on the small head-count axes). They must be listed here or
# the coverage assertion (every rank>=2 weight is mapped or allow-replicated)
# fails.
_ALLOW_REPLICATED = ("self_attention.*(query|key|value).bias",)


def _assert_qwen_shardings_and_coverage(test_case, model, layout_map):
    """Shared spec + coverage assertions for the Tier-2/3 mesh sweeps."""
    for pattern, expected in _EXPECTED_SHARDINGS.items():
        matches = [w for w in model.weights if re.search(pattern, w.path)]
        test_case.assertGreater(len(matches), 0)
        for w in matches:
            test_case.assertEqual(tuple(w.value.sharding.spec), expected)
    offending = [
        w.path
        for w in model.weights
        if len(w.shape) >= 2
        and layout_map[w.path] is None
        and not any(re.search(p, w.path) for p in _ALLOW_REPLICATED)
    ]
    test_case.assertEqual(
        offending,
        [],
        f"The following rank>=2 weights are unmapped: {offending}",
    )


class QwenBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "hidden_dim": 8,
            "intermediate_dim": 16,
            # Untied so the reverse_embeddings output-projection weight exists
            # and its layout rule is actually exercised (a tied model has no
            # such weight, making that spec assertion a dead assertion).
            "tie_word_embeddings": False,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=QwenBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 8),
            run_quantization_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=QwenBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_num_parameters(self):
        model = QwenBackbone(**self.init_kwargs)
        self.assertEqual(model.count_params(), 1384)

    def test_distribution(self):
        self.run_distribution_test(
            cls=QwenBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_shardings={
                "token_embedding/embeddings": ("model", "batch"),
                "token_embedding/reverse_embeddings": ("batch", "model"),
                "self_attention.*query.kernel": ("batch", "model", None),
                "self_attention.*key.kernel": ("batch", None, None),
                "self_attention.*value.kernel": ("batch", None, None),
                "self_attention.*attention_output.kernel": (
                    "model",
                    None,
                    "batch",
                ),
                "feedforward_intermediate_dense.kernel": ("batch", "model"),
                "feedforward_gate_dense.kernel": ("batch", "model"),
                "feedforward_output_dense.kernel": ("model", "batch"),
            },
            # q/k/v EinsumDense biases are rank-2 (num_heads, head_dim) and
            # intentionally replicated -- see _ALLOW_REPLICATED above.
            allow_replicated=("self_attention.*(query|key|value).bias",),
        )

    @parameterized.named_parameters(
        (
            f"{dims['source_preset'].split(' ')[0]}_mesh"
            f"_{'x'.join(str(s) for s in shape)}",
            dims,
            shape,
        )
        for dims in (QWEN_0_5B_DIMS, QWEN_3B_DIMS, QWEN_7B_DIMS)
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
        # matching this repo's axis_names=(..., "model") convention.
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
        layout_map = QwenBackbone.get_layout_map(device_mesh)
        distribution = keras.distribution.ModelParallel(
            layout_map=layout_map, batch_dim_name="batch"
        )
        # `head_dim` is documented in the dims dicts (as the intended scaled
        # head size) but is NOT a QwenBackbone constructor kwarg -- the model
        # derives head_dim = hidden_dim // num_query_heads -- so it is dropped
        # here along with the provenance-only "source_preset" key.
        init_kwargs = {
            k: v
            for k, v in dims.items()
            if k not in ("source_preset", "head_dim")
        }
        # Untied so reverse_embeddings is present and covered by the sweep too.
        init_kwargs["tie_word_embeddings"] = False
        with distribution.scope():
            # bfloat16: a memory mitigation for memory-constrained local
            # environments -- spec assertions are dtype-independent.
            model = QwenBackbone(dtype="bfloat16", **init_kwargs)
            _assert_qwen_shardings_and_coverage(self, model, layout_map)
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
        # (e.g. base vs instruction-tuned variants of the same size) are only
        # built once per mesh shape -- a memory/time necessity in
        # memory-constrained local environments -- while every preset in the
        # registry is still fetched and evaluated, preserving full registry
        # coverage.
        dim_keys = (
            "vocabulary_size",
            "num_query_heads",
            "num_key_value_heads",
            "hidden_dim",
            "intermediate_dim",
        )
        width_classes = {}  # dedupe key -> (config dict, [preset names])
        fetch_failures = []
        for preset in QwenBackbone.presets:
            try:
                path = get_file(preset, CONFIG_FILE)
                with open(path) as f:
                    cfg = json.load(f)["config"]
            except Exception as e:
                # A preset this account can't reach (e.g. an unaccepted
                # Kaggle license consent click-through) is logged, not
                # fatal -- the rest of the registry still gets exercised.
                fetch_failures.append((preset, str(e)))
                continue
            # num_layers is forced to 1 below regardless of the real value --
            # layout rules are per-decoder-block regexes, so depth is
            # irrelevant to spec matching/divisibility, and 1 layer keeps
            # build memory bounded in memory-constrained local environments.
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
                f"({len(fetch_failures)} fetch failures) -- likely a Kaggle "
                "license-consent gate on this account for the qwen family."
            )
        print(
            f"test_layout_map_live_presets: {len(width_classes)} unique "
            f"width-classes across {len(QwenBackbone.presets)} registry "
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
                            f"{combo_label}: needs {n_needed} devices, only "
                            f"{len(devices)} available"
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

                    # Memory-budget guard: memory-constrained local
                    # environments cannot locally build full-scale presets.
                    # Estimate this width-class's single-decoder-block bf16
                    # footprint (embedding table + untied output-projection
                    # table + one attention block's Q/K/V/O matrices + one
                    # FFN block's 3 matrices, times a 3x safety margin for
                    # JAX/XLA transient copies during construction/
                    # resharding) and skip the actual build if it exceeds a
                    # conservative local threshold. The config-fetch, dedup,
                    # and divisibility-skip logic above still exercises
                    # every registry preset either way; only the expensive
                    # build+assert step is capped.
                    hidden = cfg["hidden_dim"]
                    inter = cfg["intermediate_dim"]
                    num_kv_heads = cfg["num_key_value_heads"]
                    # Attention projections map hidden_dim<->hidden_dim (q
                    # and o), and hidden_dim<->(hidden_dim scaled by the
                    # kv/query head ratio) for GQA's k and v -- none of the
                    # four touch intermediate_dim, which is the FFN width
                    # instead.
                    kv_ratio = num_kv_heads / num_query_heads
                    est_params = (
                        # Input embedding table.
                        cfg["vocabulary_size"] * hidden
                        # Untied output-projection table (reverse_embeddings)
                        # -- this test always builds with
                        # tie_word_embeddings=False below, so the weight
                        # always exists and must be counted.
                        + cfg["vocabulary_size"] * hidden
                        + 2 * hidden * hidden  # attention q/o
                        + 2 * hidden * hidden * kv_ratio  # attention k/v
                        + 3 * hidden * inter  # FFN (gate/intermediate/output)
                    )
                    est_bytes = est_params * 2 * 3  # bf16 * safety margin
                    max_local_bytes = int(
                        os.environ.get(
                            "KERAS_HUB_DISTRIBUTION_TEST_MEM_BUDGET",
                            300 * 1024 * 1024,
                        )
                    )
                    if est_bytes > max_local_bytes:
                        reason = (
                            f"{combo_label}: estimated build memory "
                            f"~{est_bytes / 1e9:.2f}GB exceeds the "
                            f"{max_local_bytes / 1e6:.0f}MB local safety "
                            "threshold for memory-constrained local "
                            "environments -- verify this width-class on a "
                            "machine with more RAM or in CI (override via "
                            "the KERAS_HUB_DISTRIBUTION_TEST_MEM_BUDGET env "
                            "var)"
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
                    layout_map = QwenBackbone.get_layout_map(device_mesh)
                    distribution = keras.distribution.ModelParallel(
                        layout_map=layout_map, batch_dim_name="batch"
                    )
                    # `cfg` is a full serialized `get_config()` dict, which
                    # always includes a `"dtype"` key (a serialized
                    # dtype-policy dict) -- drop it so the explicit
                    # `dtype="bfloat16"` override below doesn't collide with
                    # a duplicate keyword argument. `name`/`trainable` pass
                    # through harmlessly via **kwargs.
                    init_kwargs = {k: v for k, v in cfg.items() if k != "dtype"}
                    init_kwargs["num_layers"] = 1
                    # Untied so reverse_embeddings is covered here too.
                    init_kwargs["tie_word_embeddings"] = False
                    with distribution.scope():
                        model = QwenBackbone(dtype="bfloat16", **init_kwargs)
                        _assert_qwen_shardings_and_coverage(
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
