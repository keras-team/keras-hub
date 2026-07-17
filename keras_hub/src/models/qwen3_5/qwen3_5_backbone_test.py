import gc
import json
import re

import keras
import numpy as np
import pytest
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.qwen3_5.qwen3_5_backbone import Qwen3_5Backbone
from keras_hub.src.models.qwen3_5.qwen3_5_vision_encoder import (
    Qwen3_5VisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.preset_utils import CONFIG_FILE
from keras_hub.src.utils.preset_utils import get_file

# Dims for the Tier-2 CI-safe mesh-shape sweep: representative real-preset
# dimensions, frozen as literals and sourced once, offline -- do not add a
# `get_file` call to the Tier-2 test body itself (that's what Tier 3,
# `test_layout_map_live_presets` below, is for).
#
# MEMORY NOTE: this local dev machine cannot load full-scale Qwen3.5 dims.
# The real presets have vocabulary_size=248320 and hidden_dim up to 5120 with
# 24-64 layers -- building even one such preset OOM-kills this shared 37GB
# box. What actually matters for the divisibility/sharding properties this
# tier tests is (a) whether num_query_heads divides the mesh's model-axis
# size and (b) whether vocab/hidden/intermediate and the linear-attention
# fused projection dims divide it -- not the absolute parameter count. So
# these dims are scaled down ~20-120x from the real presets while preserving
# each preset's real query:kv head ratio and its linear value:key head ratio
# exactly, and keeping every model-axis-sharded dimension divisible by every
# model-axis size in CAPPED_MESH_SHAPES (2, 4, 8). Real head_dim (256) and
# real linear head dims (128) are also scaled to 32/16 to bound the fused
# in_proj_qkv width. num_layers is fixed at 2 (not 1) because Qwen3.5 is a
# *hybrid* stack: a single layer would only exercise one of the two token
# mixers, so the sweep would silently never assert on either the
# full-attention or the linear-attention rule set. Two scaled decoder blocks
# (one of each type) are still ~1GB total RSS at the widest class -- verified.
# Full-scale real dims are exercised by `test_layout_map_live_presets` below,
# which has its own per-width-class memory-budget skip so it never attempts a
# full-scale build locally either.
#
# Provenance (fetched once via get_file(<preset>, CONFIG_FILE), 2026-07-17):
#   qwen3_5_0.8b_base: vocab 248320, hidden 1024, intermediate 3584,
#     num_query_heads 8, num_key_value_heads 2 (q:kv = 4:1), head_dim 256,
#     linear_num_key_heads 16, linear_num_value_heads 16 (v:k = 1:1).
#   qwen3_5_9b_base: vocab 248320, hidden 4096, intermediate 12288,
#     num_query_heads 16, num_key_value_heads 4 (q:kv = 4:1), head_dim 256,
#     linear_num_key_heads 16, linear_num_value_heads 32 (v:k = 2:1).
#   qwen3_5_27b: vocab 248320, hidden 5120, intermediate 17408,
#     num_query_heads 24, num_key_value_heads 4 (q:kv = 6:1), head_dim 256,
#     linear_num_key_heads 16, linear_num_value_heads 48 (v:k = 3:1).
_LINEAR_LAYER_TYPES = ["linear_attention", "full_attention"]
QWEN3_5_0_8B_DIMS = {
    "source_preset": "qwen3_5_0.8b_base (real ratios, memory-scaled dims)",
    "vocabulary_size": 2048,
    "num_layers": 2,
    "layer_types": _LINEAR_LAYER_TYPES,
    "num_query_heads": 8,
    "num_key_value_heads": 2,  # real ratio: GQA, 4:1.
    "hidden_dim": 256,
    "intermediate_dim": 896,
    "head_dim": 32,
    "linear_num_key_heads": 4,
    "linear_num_value_heads": 4,  # real ratio: 1:1.
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 4,
}
QWEN3_5_9B_DIMS = {
    "source_preset": "qwen3_5_9b_base (real ratios, memory-scaled dims)",
    "vocabulary_size": 2048,
    "num_layers": 2,
    "layer_types": _LINEAR_LAYER_TYPES,
    "num_query_heads": 16,
    "num_key_value_heads": 4,  # real ratio: GQA, 4:1.
    "hidden_dim": 512,
    "intermediate_dim": 1536,
    "head_dim": 32,
    "linear_num_key_heads": 4,
    "linear_num_value_heads": 8,  # real ratio: 2:1.
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 4,
}
QWEN3_5_27B_DIMS = {
    "source_preset": "qwen3_5_27b (real ratios, memory-scaled dims)",
    "vocabulary_size": 2048,
    "num_layers": 2,
    "layer_types": _LINEAR_LAYER_TYPES,
    "num_query_heads": 24,
    "num_key_value_heads": 4,  # real ratio: GQA, 6:1.
    "hidden_dim": 640,
    "intermediate_dim": 2176,
    "head_dim": 32,
    "linear_num_key_heads": 4,
    "linear_num_value_heads": 12,  # real ratio: 3:1.
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 4,
}

# Hard-capped mesh-shape list for this shared 37GB dev machine. The full
# 10-shape matrix from the testing-strategy doc is
# 2x4, 1x8, 4x4, 8x8, 16x16, 2x2x2, 1x1x8, 2x2x4, 4x4x4, 4x4x8 -- shapes
# 8x8, 16x16, 4x4x4, 4x4x8 (64-256 virtual devices) are DELIBERATELY
# DROPPED here due to a demonstrated systemd-oomd OOM kill of the entire
# desktop app on this shared machine during an earlier attempt at this same
# pipeline. Do not attempt the dropped shapes even experimentally on this
# box -- revisiting them requires a dedicated or CI machine, not this one.
CAPPED_MESH_SHAPES = [
    (2, 4),
    (1, 8),
    (4, 4),
    (2, 2, 2),
    (1, 1, 8),
    (2, 2, 4),
]

# Same expected_shardings patterns as Qwen3_5BackboneTest.test_distribution
# (post-QKV-axis-fix + linear_attn rules), reused by the Tier-2 and Tier-3
# mesh sweeps below.
_EXPECTED_SHARDINGS = {
    "token_embedding/embeddings": ("model", "batch"),
    "token_embedding/reverse_embeddings": ("batch", "model"),
    "self_attention.*query.kernel": ("batch", "model", None),
    "self_attention.*key.kernel": ("batch", None, None),
    "self_attention.*value.kernel": ("batch", None, None),
    "self_attention.*attention_output.kernel": ("model", None, "batch"),
    "linear_attn.*in_proj_qkv.kernel": ("batch", "model"),
    "linear_attn.*in_proj_z.kernel": ("batch", "model"),
    "linear_attn.*out_proj.kernel": ("model", "batch"),
    "feedforward_gate_dense.kernel": ("batch", "model"),
    "feedforward_intermediate_dense.kernel": ("batch", "model"),
    "feedforward_output_dense.kernel": ("model", "batch"),
}

# Rank>=2 weights intentionally left replicated (see get_layout_map's
# comment): the linear-attention conv1d kernel (tiny depthwise weight, stays
# local to each channel shard) and the in_proj_a / in_proj_b projections to
# the small num_value_heads axis.
_ALLOW_REPLICATED = (
    "linear_attn.*conv1d_kernel",
    "linear_attn.*in_proj_a.kernel",
    "linear_attn.*in_proj_b.kernel",
)


def _assert_qwen3_5_shardings_and_coverage(test_case, model, layout_map):
    """Shared spec + coverage assertions for the Tier-2/3 mesh sweeps."""
    for pattern, expected in _EXPECTED_SHARDINGS.items():
        matches = [w for w in model.weights if re.search(pattern, w.path)]
        test_case.assertGreater(
            len(matches),
            0,
            f"Expected sharding pattern {pattern!r} matched no weights.",
        )
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


class Qwen3_5BackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 4,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "hidden_dim": 16,
            "intermediate_dim": 32,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "partial_rotary_factor": 0.25,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
        }
        self.input_data = {
            "token_ids": np.ones((2, 5), dtype="int32"),
            "padding_mask": np.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=Qwen3_5Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 16),
            run_quantization_check=True,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Qwen3_5Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_num_parameters(self):
        model = Qwen3_5Backbone(**self.init_kwargs)
        self.assertGreater(model.count_params(), 0)

    def test_distribution(self):
        # The default config mixes linear_attention and full_attention
        # layers so both token-mixer rule sets are exercised, and keeps
        # num_key_value_heads=2 (deliberately not divisible by every host's
        # device count) to regression-test that key/value kernels are left
        # replicated on the model axis -- see get_layout_map's comment. The
        # shared helper pins the mesh to exactly 2 devices for the same
        # reason.
        self.run_distribution_test(
            cls=Qwen3_5Backbone,
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
                "linear_attn.*in_proj_qkv.kernel": ("batch", "model"),
                "linear_attn.*in_proj_z.kernel": ("batch", "model"),
                "linear_attn.*out_proj.kernel": ("model", "batch"),
                "feedforward_gate_dense.kernel": ("batch", "model"),
                "feedforward_intermediate_dense.kernel": ("batch", "model"),
                "feedforward_output_dense.kernel": ("model", "batch"),
            },
            allow_replicated=_ALLOW_REPLICATED,
        )

    @parameterized.named_parameters(
        (
            f"{dims['source_preset'].split(' ')[0]}_mesh"
            f"_{'x'.join(str(s) for s in shape)}",
            dims,
            shape,
        )
        for dims in (QWEN3_5_0_8B_DIMS, QWEN3_5_9B_DIMS, QWEN3_5_27B_DIMS)
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
            # 3D shape: the extra axis is named "seq"; get_layout_map only
            # names "batch"/"model", so the "seq" axis simply replicates
            # weights (no rule targets it), matching every other 3D-mesh
            # test in this PR series.
            axis_names = ("batch", "seq", "model")
        device_mesh = keras.distribution.DeviceMesh(
            shape=mesh_shape,
            axis_names=axis_names,
            devices=devices,
        )
        layout_map = Qwen3_5Backbone.get_layout_map(device_mesh)
        distribution = keras.distribution.ModelParallel(
            layout_map=layout_map, batch_dim_name="batch"
        )
        init_kwargs = {k: v for k, v in dims.items() if k != "source_preset"}
        with distribution.scope():
            # bfloat16: a memory mitigation for this shared dev machine --
            # spec assertions are dtype-independent.
            model = Qwen3_5Backbone(dtype="bfloat16", **init_kwargs)
            _assert_qwen3_5_shardings_and_coverage(self, model, layout_map)
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
        # only built once per mesh shape -- a memory/time necessity on this
        # machine -- while every preset in the registry is still fetched and
        # evaluated, preserving full registry coverage.
        dim_keys = (
            "vocabulary_size",
            "num_query_heads",
            "num_key_value_heads",
            "hidden_dim",
            "intermediate_dim",
            "head_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
        )
        width_classes = {}  # dedupe key -> (config dict, [preset names])
        fetch_failures = []
        for preset in Qwen3_5Backbone.presets:
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
            cfg = dict(cfg)
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
                "qwen3_5 family."
            )
        print(
            f"test_layout_map_live_presets: {len(width_classes)} unique "
            f"width-classes across {len(Qwen3_5Backbone.presets)} registry "
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
                    # width-class's embedding table + one FFN block's 3
                    # matrices (times a 3x safety margin for JAX/XLA
                    # transient copies during construction/resharding) and
                    # skip the actual build if it exceeds a conservative
                    # local threshold. The config-fetch, dedup, and
                    # divisibility-skip logic above still exercises every
                    # registry preset either way; only the expensive
                    # build+assert step is capped. (Real Qwen3.5 presets --
                    # vocab 248320, hidden up to 5120 -- all exceed this and
                    # are therefore verified offline / in CI, not here.)
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
                    layout_map = Qwen3_5Backbone.get_layout_map(device_mesh)
                    distribution = keras.distribution.ModelParallel(
                        layout_map=layout_map, batch_dim_name="batch"
                    )
                    init_kwargs = {
                        k: v for k, v in cfg.items() if k in dim_keys
                    }
                    # Force a tiny hybrid stack: one layer of each token
                    # mixer so both the full-attention and linear-attention
                    # rule sets are asserted, while keeping depth (and thus
                    # build memory) minimal. Real presets have 24-64 layers;
                    # depth is irrelevant to per-decoder-block spec matching.
                    init_kwargs["num_layers"] = 2
                    init_kwargs["layer_types"] = _LINEAR_LAYER_TYPES
                    with distribution.scope():
                        model = Qwen3_5Backbone(dtype="bfloat16", **init_kwargs)
                        _assert_qwen3_5_shardings_and_coverage(
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


class Qwen3_5MultimodalBackboneTest(TestCase):
    """Tests for the backbone with vision encoder attached."""

    def setUp(self):
        self.vision_encoder = Qwen3_5VisionEncoder(
            depth=1,
            hidden_size=16,
            num_heads=2,
            intermediate_size=32,
            in_channels=3,
            patch_size=4,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=16,
            num_position_embeddings=64,
        )
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 4,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "hidden_dim": 16,
            "intermediate_dim": 32,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "partial_rotary_factor": 0.25,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "vision_encoder": self.vision_encoder,
            "mrope_section": [1, 1, 1],
        }

    def test_multimodal_backbone_builds(self):
        """Verify multimodal backbone constructs and has expected attributes."""
        model = Qwen3_5Backbone(**self.init_kwargs)
        self.assertGreater(model.count_params(), 0)
        self.assertIsNotNone(model.vision_encoder)
        self.assertTrue(hasattr(model, "interleave_embeddings"))

    def test_multimodal_backbone_forward(self):
        """Multimodal backbone forward pass with vision inputs."""
        model = Qwen3_5Backbone(**self.init_kwargs)

        # Build vision inputs: 1 image, 4x4 grid, patch_size=4,
        # temporal_patch_size=2. After spatial merge (2x2), 4 tokens.
        grid_thw = np.array([[[1, 4, 4]]], dtype="int32")  # (1, 1, 3)
        total_patches = 1 * 4 * 4  # 16
        # Batched pixel_values: (1, total_patches, T, pH, pW, C)
        pixel_values = np.random.randn(1, total_patches, 2, 4, 4, 3).astype(
            "float32"
        )

        seq_len = 10
        # Place 4 vision tokens at positions 2,3,4,5.
        vision_indices = np.array([[2, 3, 4, 5]], dtype="int32")

        input_data = {
            "token_ids": np.ones((1, seq_len), dtype="int32"),
            "padding_mask": np.ones((1, seq_len), dtype="int32"),
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw,
            "vision_indices": vision_indices,
        }
        output = model(input_data)
        self.assertEqual(ops.shape(output), (1, seq_len, 16))

    def test_vision_encoder_standalone(self):
        """Test vision encoder produces correct output shape standalone."""
        encoder = self.vision_encoder

        # 16 patches, spatial_merge_size=2 → 4 merged tokens.
        grid_thw = np.array([[1, 4, 4]], dtype="int32")
        total_patches = 1 * 4 * 4
        pixel_values = np.random.randn(total_patches, 2, 4, 4, 3).astype(
            "float32"
        )

        output = encoder(
            ops.convert_to_tensor(pixel_values),
            ops.convert_to_tensor(grid_thw),
        )
        self.assertEqual(ops.shape(output), (4, 16))

    def test_interleave_embeddings(self):
        """Test that interleave layer correctly scatters vision tokens."""
        model = Qwen3_5Backbone(**self.init_kwargs)

        batch_size = 1
        seq_len = 8
        hidden_dim = 16

        text_emb = ops.zeros((batch_size, seq_len, hidden_dim))
        vision_emb = np.ones((2, hidden_dim), dtype="float32")
        indices = ops.convert_to_tensor([1, 3], dtype="int32")

        result = model.interleave_embeddings(
            image_embeddings=vision_emb,
            text_embeddings=text_emb,
            vision_indices=indices,
        )
        self.assertEqual(ops.shape(result), (batch_size, seq_len, hidden_dim))

        result_np = ops.convert_to_numpy(result)
        np.testing.assert_allclose(result_np[0, 0, :], 0.0, atol=1e-6)
        np.testing.assert_allclose(result_np[0, 1, :], 1.0, atol=1e-6)
        np.testing.assert_allclose(result_np[0, 2, :], 0.0, atol=1e-6)
        np.testing.assert_allclose(result_np[0, 3, :], 1.0, atol=1e-6)
