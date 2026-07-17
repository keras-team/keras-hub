import gc
import json
import os
import re

import keras
import pytest
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.gemma.gemma_backbone import GemmaBackbone
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.preset_utils import CONFIG_FILE
from keras_hub.src.utils.preset_utils import get_file

# Dims for the Tier-2 CI-safe mesh-shape sweep: representative real-preset
# dimensions, frozen as literals and sourced once, offline -- do not add a
# `get_file` call to the Tier-2 test body itself (that's what Tier 3,
# `test_layout_map_live_presets` below, is for).
#
# MEMORY NOTE: memory-constrained local environments cannot load full-scale
# model dims -- an earlier attempt using literal preset dims (e.g. Gemma-2
# 27B-class: hidden 4608 / intermediate 73728 / vocab 256128, even at
# num_layers=1) drove a single test process to ~23GB RSS across the 18
# parameterized cases and triggered an OOM kill (see CAPPED_MESH_SHAPES
# comment for the mesh-size OOM history; this was a second, distinct OOM
# from dimension size, not mesh size). What actually matters for the
# divisibility/sharding properties this tier tests is the RATIO of query
# heads to kv heads and whether hidden/intermediate/vocab divide the mesh's
# model-axis sizes -- not the absolute parameter count. So these dims are
# scaled down by roughly 20-30x from the real presets while preserving each
# preset's real query:kv head ratio (8:1 MQA, 16:1 MQA, 32:16 GQA) and
# keeping hidden/intermediate/vocab as clean powers of 2 divisible by every
# mesh shape in CAPPED_MESH_SHAPES. Full-scale real dims are exercised by
# `test_layout_map_live_presets` below, which has its own per-width-class
# memory-budget skip so it never attempts a full-scale build in a
# memory-constrained environment either -- true full-scale verification
# happens on a machine with more RAM or in CI (Tier 4 in the design doc).
GEMMA_2B_DIMS = {
    "source_preset": "gemma_2b_en (real ratio, memory-scaled dims)",
    "vocabulary_size": 2048,
    "num_layers": 1,  # depth is irrelevant to spec matching/divisibility.
    "num_query_heads": 8,
    "num_key_value_heads": 1,  # real ratio: MQA, 8:1.
    "hidden_dim": 256,
    "intermediate_dim": 1024,
    "head_dim": 32,
}
GEMMA_7B_DIMS = {
    "source_preset": "gemma_7b_en (real ratio, memory-scaled dims)",
    "vocabulary_size": 2048,
    "num_layers": 1,
    "num_query_heads": 16,
    "num_key_value_heads": 1,  # real ratio: MQA, 16:1.
    "hidden_dim": 384,
    "intermediate_dim": 1536,
    "head_dim": 32,
}
GEMMA2_9B_DIMS = {
    "source_preset": "gemma2_9b_en (real ratio, memory-scaled dims)",
    "vocabulary_size": 2048,
    "num_layers": 1,
    "num_query_heads": 32,
    "num_key_value_heads": 16,  # real ratio: GQA, 2:1.
    "hidden_dim": 512,
    "intermediate_dim": 2048,
    "head_dim": 32,
    # Real Gemma-2 architecture flags -- without these this "Gemma2" config
    # would silently build with Gemma-1 defaults (both False) instead.
    "use_post_ffw_norm": True,
    "use_post_attention_norm": True,
    "attention_logit_soft_cap": 50.0,
    "final_logit_soft_cap": 30.0,
    "use_sliding_window_attention": True,
    "sliding_window_size": 4096,
}

# Hard-capped mesh-shape list for memory-constrained local environments.
# The full 10-shape matrix from the testing-strategy doc is
# 2x4, 1x8, 4x4, 8x8, 16x16, 2x2x2, 1x1x8, 2x2x4, 4x4x4, 4x4x8 -- shapes
# 8x8, 16x16, 4x4x4, 4x4x8 (64-256 virtual devices) are DELIBERATELY
# DROPPED here: they exceed a typical memory-constrained local environment's
# memory budget (a demonstrated systemd-oomd OOM kill of the entire desktop
# app during an earlier attempt at this same pipeline, confirmed via
# journalctl). Do not attempt the dropped shapes experimentally in such an
# environment -- revisiting them requires a dedicated or CI machine with
# more memory.
CAPPED_MESH_SHAPES = [
    (2, 4),
    (1, 8),
    (4, 4),
    (2, 2, 2),
    (1, 1, 8),
    (2, 2, 4),
]

# Same 8 expected_shardings patterns as GemmaBackboneTest.test_distribution
# (post-kv-fix), reused by the Tier-2 and Tier-3 mesh sweeps below.
_EXPECTED_SHARDINGS = {
    "token_embedding/embeddings": ("model", "batch"),
    "attention/query/kernel": ("model", "batch", None),
    "attention/key/kernel": (None, "batch", None),
    "attention/value/kernel": (None, "batch", None),
    "attention/attention_output/kernel": ("model", None, "batch"),
    "ffw_gating/kernel": ("batch", "model"),
    "ffw_gating_2/kernel": ("batch", "model"),
    "ffw_linear.*kernel": ("model", "batch"),
}


def _assert_gemma_shardings_and_coverage(test_case, model, layout_map):
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


class GemmaBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 20,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 1,
            "hidden_dim": 16,
            "intermediate_dim": 32,
            "head_dim": 4,
            "layer_norm_epsilon": 1e-6,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=GemmaBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 16),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=GemmaBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_smallest_preset(self):
        # TODO: Fails with OOM on current GPU CI
        self.run_preset_test(
            cls=GemmaBackbone,
            preset="gemma_2b_en",
            input_data={
                "token_ids": ops.array([[651, 4320, 8426, 25341, 235265]]),
                "padding_mask": ops.ones((1, 5), dtype="int32"),
            },
            expected_output_shape=(1, 5, 2048),
            # The forward pass from a preset should be stable!
            expected_partial_output=ops.array(
                [1.073359, 0.262374, 0.170238, 0.605402, 2.336161]
            ),
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in GemmaBackbone.presets:
            self.run_preset_test(
                cls=GemmaBackbone,
                preset=preset,
                input_data=self.input_data,
            )

    def test_architecture_characteristics(self):
        model = GemmaBackbone(**self.init_kwargs)
        self.assertEqual(model.count_params(), 3216)
        self.assertEqual(len(model.layers), 6)

    @pytest.mark.multi_device
    def test_distribution(self):
        # Note (preserved from the pre-refactor manual test): mesh is
        # pinned to exactly 2 devices (not len(devices), see the shared
        # helper) so that the default test config's num_key_value_heads=1
        # -- intentionally not divisible by every host's device count --
        # regression-tests that key/value kernels are left replicated
        # rather than sharded. See get_layout_map's comment for why.
        self.run_distribution_test(
            cls=GemmaBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_shardings={
                "token_embedding/embeddings": ("model", "batch"),
                "attention/query/kernel": ("model", "batch", None),
                "attention/key/kernel": (None, "batch", None),
                "attention/value/kernel": (None, "batch", None),
                "attention/attention_output/kernel": (
                    "model",
                    None,
                    "batch",
                ),
                "ffw_gating/kernel": ("batch", "model"),
                "ffw_gating_2/kernel": ("batch", "model"),
                "ffw_linear.*kernel": ("model", "batch"),
            },
            allow_replicated=(),
        )

    @pytest.mark.multi_device
    def test_distribution_with_lora(self):
        if keras.backend.backend() != "jax":
            self.skipTest("`ModelParallel` testing requires the Jax backend.")
        devices = keras.distribution.list_devices("CPU")
        if len(devices) < 2:
            self.skipTest("`ModelParallel` testing requires multiple devices.")
        # Pinned to exactly 2 devices -- see test_distribution's comment.
        devices = devices[:2]
        device_mesh = keras.distribution.DeviceMesh(
            shape=(1, 2),
            axis_names=("batch", "model"),
            devices=devices,
        )

        layout_map = GemmaBackbone.get_layout_map(device_mesh)
        distribution = keras.distribution.ModelParallel(layout_map=layout_map)
        with distribution.scope():
            model = GemmaBackbone(**self.init_kwargs)
            model.enable_lora(rank=4)

        for w in model.weights:
            if "attention/query/lora_kernel_a" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), (None, None, None)
                )
            if "attention/query/lora_kernel_b" in w.path:
                self.assertEqual(tuple(w.value.sharding.spec), (None, None))
            if "attention/value/lora_kernel_a" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), (None, None, None)
                )
            if "attention/value/lora_kernel_b" in w.path:
                self.assertEqual(tuple(w.value.sharding.spec), (None, None))

    @parameterized.named_parameters(
        (
            f"{dims['source_preset'].split(' ')[0]}_mesh"
            f"_{'x'.join(str(s) for s in shape)}",
            dims,
            shape,
        )
        for dims in (GEMMA_2B_DIMS, GEMMA_7B_DIMS, GEMMA2_9B_DIMS)
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
        layout_map = GemmaBackbone.get_layout_map(device_mesh)
        distribution = keras.distribution.ModelParallel(
            layout_map=layout_map, batch_dim_name="batch"
        )
        init_kwargs = {k: v for k, v in dims.items() if k != "source_preset"}
        with distribution.scope():
            # bfloat16: a memory mitigation for memory-constrained local
            # environments -- spec assertions are dtype-independent.
            model = GemmaBackbone(dtype="bfloat16", **init_kwargs)
            _assert_gemma_shardings_and_coverage(self, model, layout_map)
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
        # (e.g. base vs instruction-tuned variants of the same size) are
        # only built once per mesh shape -- a memory/time necessity in
        # memory-constrained local environments (gemma2_27b-width builds
        # are ~2.5GB even at 1 layer bf16), while every preset in the
        # registry is still fetched and evaluated, preserving full registry
        # coverage.
        dim_keys = (
            "vocabulary_size",
            "num_query_heads",
            "num_key_value_heads",
            "hidden_dim",
            "intermediate_dim",
            "head_dim",
        )
        width_classes = {}  # dedupe key -> (config dict, [preset names])
        fetch_failures = []
        for preset in GemmaBackbone.presets:
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
            # num_layers is forced to 1 below regardless of the real
            # value -- layout rules are per-decoder-block regexes, so
            # depth is irrelevant to spec matching/divisibility, and 1
            # layer keeps build memory bounded in memory-constrained local
            # environments.
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
                "gemma family. See the module comment above "
                "CAPPED_MESH_SHAPES."
            )
        print(
            f"test_layout_map_live_presets: {len(width_classes)} unique "
            f"width-classes across {len(GemmaBackbone.presets)} registry "
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

                    # Memory-budget guard: memory-constrained local
                    # environments cannot locally build full-scale presets
                    # (a prior attempt at gemma2_27b-class dims alone drove
                    # one process to ~23GB RSS and triggered an OOM kill --
                    # see CAPPED_MESH_SHAPES' comment). Estimate this
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
                    # Tunable via env var so CI or a bigger machine can opt
                    # into real full-scale verification; defaults to 300MB
                    # to preserve today's behavior on memory-constrained
                    # local environments.
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
                            "machine with more RAM or in CI (override with "
                            "KERAS_HUB_DISTRIBUTION_TEST_MEM_BUDGET)"
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
                    layout_map = GemmaBackbone.get_layout_map(device_mesh)
                    distribution = keras.distribution.ModelParallel(
                        layout_map=layout_map, batch_dim_name="batch"
                    )
                    # Use the full preset config (already num_layers=1'd
                    # above), not just dim_keys -- filtering to dim_keys
                    # here would discard architecture flags like
                    # use_post_attention_norm/use_post_ffw_norm/
                    # attention_logit_soft_cap, silently building every
                    # Gemma-2-family preset with Gemma-1 defaults instead.
                    # cfg is the preset's serialized `config` dict, which
                    # keras's get_config()/from_config() convention already
                    # restricts to valid constructor kwargs.
                    init_kwargs = dict(cfg)
                    with distribution.scope():
                        model = GemmaBackbone(dtype="bfloat16", **init_kwargs)
                        _assert_gemma_shardings_and_coverage(
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


class Gemma2BackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 20,  # 256128
            "num_layers": 2,  # 46
            "num_query_heads": 4,  # 32
            "num_key_value_heads": 2,  # 16
            "hidden_dim": 16,  # 4608
            "intermediate_dim": 32,  # 73728
            "head_dim": 4,  # 128
            "sliding_window_size": 5,  # 4096
            "attention_logit_soft_cap": 50,
            "final_logit_soft_cap": 30,
            "layer_norm_epsilon": 1e-6,
            "query_head_dim_normalize": False,
            "use_post_ffw_norm": True,
            "use_post_attention_norm": True,
            "use_sliding_window_attention": True,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 10), dtype="int32"),
            "padding_mask": ops.ones((2, 10), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=GemmaBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 10, 16),
        )

    def test_sliding_window(self):
        # Test sliding window correctness by hand.
        backbone = GemmaBackbone(**self.init_kwargs)
        attention = backbone.transformer_layers[0].attention
        mask = attention._mask_sliding_window(ops.ones((1, 10, 10), "bool"))
        expected = [
            [
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            ]
        ]
        self.assertAllEqual(mask, expected)

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=GemmaBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
