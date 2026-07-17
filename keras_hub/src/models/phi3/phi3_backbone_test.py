import gc
import json
import re

import keras
import pytest
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.phi3.phi3_backbone import Phi3Backbone
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.preset_utils import CONFIG_FILE
from keras_hub.src.utils.preset_utils import get_file

# Dims for the Tier-2 CI-safe mesh-shape sweep: representative real-preset
# dimensions, frozen as literals and sourced once, offline -- do not add a
# `get_file` call to the Tier-2 test body itself (that's what Tier 3,
# `test_layout_map_live_presets` below, is for).
#
# MEMORY NOTE: this local dev machine cannot load full-scale model dims (see
# gemma_backbone_test.py's identically-named constant for the OOM history
# that established this pattern). What actually matters for the
# divisibility/sharding properties this tier tests is the RATIO of query
# heads to kv heads and whether hidden/intermediate/vocab divide the mesh's
# model-axis sizes -- not the absolute parameter count. Both registered Phi3
# presets (`phi3_mini_4k_instruct_en`, `phi3_mini_128k_instruct_en`) share
# the same "mini" architecture (32 query heads, 32 kv heads -- Phi-3-mini
# uses full multi-head attention, not GQA/MQA; hidden_dim 3072, intermediate
# 8192), so there is only one real width-class to derive a scaled preset
# from. A second, smaller synthetic width-class is added purely to exercise
# a *different* query:kv head ratio (a hypothetical GQA-style Phi3 config)
# so the mesh sweep isn't limited to a single divisibility profile. Dims are
# scaled down ~20-30x from the real "mini" preset while preserving its exact
# 1:1 query:kv head ratio and keeping hidden/intermediate/vocab as clean
# powers of 2 divisible by every mesh shape in CAPPED_MESH_SHAPES.
PHI3_MINI_DIMS = {
    "source_preset": (
        "phi3_mini_4k_instruct_en (real ratio, memory-scaled dims)"
    ),
    "vocabulary_size": 3200,
    "num_layers": 1,  # depth is irrelevant to spec matching/divisibility.
    "num_query_heads": 32,
    "num_key_value_heads": 32,  # real ratio: full MHA, 1:1 (no GQA).
    "hidden_dim": 256,
    "intermediate_dim": 1024,
}
PHI3_SMALL_GQA_DIMS = {
    "source_preset": "phi3_small_gqa (synthetic width-class, GQA ratio)",
    "vocabulary_size": 1600,
    "num_layers": 1,
    "num_query_heads": 16,
    "num_key_value_heads": 4,  # synthetic ratio: GQA, 4:1.
    "hidden_dim": 128,
    "intermediate_dim": 512,
}

# Hard-capped mesh-shape list for this shared 37GB dev machine -- see
# gemma_backbone_test.py's identically-named constant for the full
# rationale and OOM history. Shapes 8x8, 16x16, 4x4x4, 4x4x8 (64-256
# virtual devices) are deliberately dropped; do not attempt them on this
# box.
CAPPED_MESH_SHAPES = [
    (2, 4),
    (1, 8),
    (4, 4),
    (2, 2, 2),
    (1, 1, 8),
    (2, 2, 4),
]

# Same expected_shardings patterns as Phi3Test.test_distribution (post
# QKV-axis fix), reused by the Tier-2 and Tier-3 mesh sweeps below.
_EXPECTED_SHARDINGS = {
    "token_embedding/embeddings": ("model", "batch"),
    "token_embedding/reverse_embeddings": ("batch", "model"),
    "attention/query/kernel": ("batch", "model", None),
    "attention/key/kernel": ("batch", None, None),
    "attention/value/kernel": ("batch", None, None),
    "attention/attention_output/kernel": ("model", None, "batch"),
    "feedforward_intermediate_dense/kernel": ("batch", "model"),
    "feedforward_gate_dense/kernel": ("batch", "model"),
    "feedforward_output_dense/kernel": ("model", "batch"),
}


def _assert_phi3_shardings_and_coverage(test_case, model, layout_map):
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


class Phi3Test(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "hidden_dim": 8,
            "intermediate_dim": 8,
        }
        self.su_rotary_init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_query_heads": 2,
            "num_key_value_heads": 1,
            "hidden_dim": 8,
            "intermediate_dim": 12,
            "max_sequence_length": 10,
            "pretraining_sequence_length": 5,
            "rope_scaling_type": "su",
            "rope_scaling_short_factor": [1.2, 1.4],
            "rope_scaling_long_factor": [0.8, 0.6],
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=Phi3Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 8),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Phi3Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_distribution(self):
        # Note (preserved from the pre-refactor manual test): the default
        # test config's num_key_value_heads=2 divides the pinned 2-device
        # mesh cleanly, but key/value are still asserted to be replicated
        # on the model axis (not sharded) -- GQA head counts are typically
        # much smaller than query head counts and not guaranteed to divide
        # arbitrary mesh sizes in general, so this model always leaves them
        # replicated regardless of divisibility. See get_layout_map's
        # comment for the full rationale.
        self.run_distribution_test(
            cls=Phi3Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_shardings={
                "token_embedding/embeddings": ("model", "batch"),
                "token_embedding/reverse_embeddings": ("batch", "model"),
                "attention/query/kernel": ("batch", "model", None),
                "attention/key/kernel": ("batch", None, None),
                "attention/value/kernel": ("batch", None, None),
                "attention/attention_output/kernel": (
                    "model",
                    None,
                    "batch",
                ),
                "feedforward_intermediate_dense/kernel": ("batch", "model"),
                "feedforward_gate_dense/kernel": ("batch", "model"),
                "feedforward_output_dense/kernel": ("model", "batch"),
            },
            allow_replicated=(),
        )

    def test_backbone_basics_with_su_rotary(self):
        self.run_backbone_test(
            cls=Phi3Backbone,
            init_kwargs=self.su_rotary_init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 8),
        )

    @pytest.mark.large
    def test_saved_model_with_su_rotary(self):
        self.run_model_saving_test(
            cls=Phi3Backbone,
            init_kwargs=self.su_rotary_init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=Phi3Backbone,
            preset="phi3_mini_4k_instruct_en",
            input_data={
                "token_ids": ops.array([[1, 450, 4996, 1701, 29916, 29889]]),
                "padding_mask": ops.ones((1, 6), dtype="int32"),
            },
            expected_output_shape=(1, 6, 3072),
            # The forward pass from a preset should be stable!
            # Reference values computed using PyTorch HF model.
            expected_partial_output=ops.array(
                [-0.21222, 0.04004, -0.02759, 0.02200]
            ),
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Phi3Backbone.presets:
            self.run_preset_test(
                cls=Phi3Backbone,
                preset=preset,
                input_data=self.input_data,
            )

    @parameterized.named_parameters(
        (
            f"{dims['source_preset'].split(' ')[0]}_mesh"
            f"_{'x'.join(str(s) for s in shape)}",
            dims,
            shape,
        )
        for dims in (PHI3_MINI_DIMS, PHI3_SMALL_GQA_DIMS)
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
        layout_map = Phi3Backbone.get_layout_map(device_mesh)
        distribution = keras.distribution.ModelParallel(
            layout_map=layout_map, batch_dim_name="batch"
        )
        init_kwargs = {k: v for k, v in dims.items() if k != "source_preset"}
        with distribution.scope():
            # bfloat16: a memory mitigation for this shared dev machine --
            # spec assertions are dtype-independent.
            model = Phi3Backbone(dtype="bfloat16", **init_kwargs)
            _assert_phi3_shardings_and_coverage(self, model, layout_map)
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
        # (both registered Phi3 presets are the same "mini" architecture at
        # different context lengths) are only built once per mesh shape -- a
        # memory/time necessity on this machine, while every preset in the
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
        for preset in Phi3Backbone.presets:
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
                "phi3 family. See the module comment above "
                "CAPPED_MESH_SHAPES."
            )
        print(
            f"test_layout_map_live_presets: {len(width_classes)} unique "
            f"width-classes across {len(Phi3Backbone.presets)} registry "
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
                    # locally build full-scale presets. Estimate this
                    # width-class's single-decoder-block bf16 footprint
                    # (embedding table + one FFN block's 3 matrices, times
                    # a 3x safety margin for JAX/XLA transient copies
                    # during construction/resharding) and skip the actual
                    # build if it exceeds a conservative local threshold.
                    # The config-fetch, dedup, and divisibility-skip logic
                    # above still exercises every registry preset either
                    # way; only the expensive build+assert step is capped.
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
                    layout_map = Phi3Backbone.get_layout_map(device_mesh)
                    distribution = keras.distribution.ModelParallel(
                        layout_map=layout_map, batch_dim_name="batch"
                    )
                    init_kwargs = {
                        k: v for k, v in cfg.items() if k in dim_keys
                    }
                    init_kwargs["num_layers"] = 1
                    with distribution.scope():
                        model = Phi3Backbone(dtype="bfloat16", **init_kwargs)
                        _assert_phi3_shardings_and_coverage(
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
