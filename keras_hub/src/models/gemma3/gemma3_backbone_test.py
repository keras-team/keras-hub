import copy
import gc
import json
import os
import re

import keras
import numpy as np
import pytest
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
from keras_hub.src.models.gemma3.gemma3_vision_encoder import (
    Gemma3VisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.preset_utils import CONFIG_FILE
from keras_hub.src.utils.preset_utils import get_file

# Text-decoder expected shardings (post-fix), shared by test_distribution and
# the Tier-2/Tier-3 mesh sweeps. Gemma3's attention kernels use the Gemma
# `(num_heads, hidden, head_dim)` convention (einsum `btd,ndh->btnh`), so
# heads-on-model (`query = ("model", "batch", None)`) is already the
# communication-efficient Megatron column-parallel choice and stays unchanged
# from merged code -- only the vision tower is newly sharded (see B.2).
_TEXT_EXPECTED_SHARDINGS = {
    "token_embedding/embeddings": ("model", "batch"),
    "decoder_block.*attention/query/kernel": ("model", "batch", None),
    "decoder_block.*attention/key/kernel": (None, "batch", None),
    "decoder_block.*attention/value/kernel": (None, "batch", None),
    "decoder_block.*attention_output/kernel": ("model", None, "batch"),
    "decoder_block.*ffw_gating/kernel": ("batch", "model"),
    "decoder_block.*ffw_gating_2/kernel": ("batch", "model"),
    "decoder_block.*ffw_linear/kernel": ("model", "batch"),
}

# Dims for the Tier-2 CI-safe mesh-shape sweep: representative real-preset
# dimensions, frozen as literals and sourced once, offline -- do not add a
# `get_file` call to the Tier-2 test body itself (that is what Tier 3,
# `test_layout_map_live_presets` below, is for).
#
# MEMORY NOTE: memory-constrained local environments cannot load full-scale
# model dims. What actually matters for the divisibility/sharding properties
# this tier tests is the RATIO of query heads to kv heads and whether
# hidden/intermediate/vocab divide the mesh's model-axis sizes -- not the
# absolute parameter count. So these dims are scaled down by roughly 20-30x
# from the real gemma3 presets while preserving each preset's real query:kv
# head ratio (gemma3_1b 4:1 GQA, gemma3_4b 8:4 GQA, gemma3_27b 32:16 GQA) and
# keeping hidden/intermediate/vocab as clean multiples divisible by every
# mesh shape in CAPPED_MESH_SHAPES. Full-scale real dims are exercised by
# `test_layout_map_live_presets` below, which has its own per-width-class
# memory-budget skip so it never attempts a full-scale build in a
# memory-constrained environment either -- true full-scale verification
# happens on a machine with more RAM or in CI.
GEMMA3_1B_DIMS = {
    "source_preset": "gemma3_1b (real ratio, memory-scaled dims)",
    "vocabulary_size": 2048,  # real 262144
    "image_size": 16,
    "num_layers": 1,  # depth is irrelevant to spec matching/divisibility.
    "num_query_heads": 4,
    "num_key_value_heads": 1,  # real ratio: GQA, 4:1.
    "hidden_dim": 256,  # real 1152
    "intermediate_dim": 512,  # real 6912
    "head_dim": 32,  # real 256
}
GEMMA3_4B_DIMS = {
    "source_preset": "gemma3_4b (real ratio, memory-scaled dims)",
    "vocabulary_size": 2048,  # real 262144
    "image_size": 16,
    "num_layers": 1,
    "num_query_heads": 8,
    "num_key_value_heads": 4,  # real ratio: GQA, 2:1.
    "hidden_dim": 384,  # real 2560
    "intermediate_dim": 1024,  # real 10240
    "head_dim": 32,  # real 256
}
GEMMA3_27B_DIMS = {
    "source_preset": "gemma3_27b (real ratio, memory-scaled dims)",
    "vocabulary_size": 2048,  # real 262144
    "image_size": 16,
    "num_layers": 1,
    "num_query_heads": 32,
    "num_key_value_heads": 16,  # real ratio: GQA, 2:1.
    "hidden_dim": 512,  # real 5376
    "intermediate_dim": 2048,  # real 21504
    "head_dim": 32,  # real 128
}

# Hard-capped mesh-shape list for memory-constrained local environments. The
# full 10-shape matrix from the testing-strategy doc is
# 2x4, 1x8, 4x4, 8x8, 16x16, 2x2x2, 1x1x8, 2x2x4, 4x4x4, 4x4x8 -- shapes
# 8x8, 16x16, 4x4x4, 4x4x8 (64-256 virtual devices) are DELIBERATELY DROPPED
# here: they exceed a typical memory-constrained local environment's memory
# budget (a prior attempt at this pipeline was OOM-killed). Do not attempt
# the dropped shapes experimentally in such an environment -- revisiting them
# requires a dedicated or CI machine with more memory.
CAPPED_MESH_SHAPES = [
    (2, 4),
    (1, 8),
    (4, 4),
    (2, 2, 2),
    (1, 1, 8),
    (2, 2, 4),
]


def _assert_text_shardings_and_coverage(test_case, model, layout_map):
    """Shared spec + coverage assertions for the Tier-2/3 text mesh sweeps."""
    for pattern, expected in _TEXT_EXPECTED_SHARDINGS.items():
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


class Gemma3BackboneTest(TestCase, parameterized.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.vocabulary_size = 256
        self.text_sequence_length = 64
        self.image_size = 16
        self.vision_tokens_per_image = int((self.image_size / 4) ** 2)
        self.max_images_per_prompt = 3

        # Small, in-scope-constructible vision-encoder kwargs. Kept as a dict
        # (not a prebuilt instance) so the distribution test can build the
        # encoder *inside* `distribution.scope()` -- see `vision_init_kwargs`.
        self.vision_encoder_kwargs = {
            "image_size": self.image_size,
            "patch_size": 4,
            "pool_size": 2,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 8,
            "intermediate_dim": 16,
            "output_dim": 8,
        }

        # === Vision + Text Backbone ===
        vision_encoder = Gemma3VisionEncoder(**self.vision_encoder_kwargs)

        self.init_kwargs = {
            # vocabulary
            "vocabulary_size": self.vocabulary_size,
            # image
            "image_size": self.image_size,
            # model
            "num_layers": 6,
            "num_query_heads": 2,
            "num_key_value_heads": 1,
            "hidden_dim": 8,
            "intermediate_dim": 16,
            "head_dim": 4,
            # other model args
            "query_head_dim_normalize": True,
            "use_query_key_norm": True,
            "use_post_ffw_norm": True,
            "use_post_attention_norm": True,
            "final_logit_soft_cap": None,
            "attention_logit_soft_cap": None,
            "use_sliding_window_attention": True,
            "sliding_window_size": 1024,
            "vision_encoder": vision_encoder,
        }

        dummy_images = np.random.rand(
            self.batch_size,
            self.max_images_per_prompt,
            self.image_size,
            self.image_size,
            3,
        )
        dummy_text_token_ids = np.random.rand(
            self.batch_size, self.text_sequence_length
        )

        self.input_data = {
            "token_ids": dummy_text_token_ids,
            "images": dummy_images,
            "padding_mask": np.ones(
                (self.batch_size, self.text_sequence_length),
                dtype="int32",
            ),
        }
        vision_mask_0 = [False] * 20 + [True] * 8 + [False] * 32 + [True] * 4
        vision_mask_1 = [False] * 16 + [True] * 8 + [False] * 36 + [True] * 4
        self.input_data["vision_mask"] = np.array(
            [vision_mask_0, vision_mask_1]
        )
        self.input_data["vision_indices"] = np.array(
            [
                list(range(20, 28)) + list(range(60, 64)),
                list(range(16, 24)) + list(range(60, 64)),
            ]
        )

        # === Text Backbone ===
        self.text_init_kwargs = copy.deepcopy(self.init_kwargs)
        del self.text_init_kwargs["vision_encoder"]

        self.text_backbone_input_data = copy.deepcopy(self.input_data)
        del self.text_backbone_input_data["images"]
        del self.text_backbone_input_data["vision_mask"]
        del self.text_backbone_input_data["vision_indices"]

    @parameterized.named_parameters(
        ("text_and_vision", "text_and_vision"), ("text_only", "text_only")
    )
    def test_backbone_basics(self, backbone_type):
        if backbone_type == "text_and_vision":
            init_kwargs = self.init_kwargs
            input_data = self.input_data
        elif backbone_type == "text_only":
            init_kwargs = self.text_init_kwargs
            input_data = self.text_backbone_input_data

        self.run_backbone_test(
            cls=Gemma3Backbone,
            init_kwargs=init_kwargs,
            input_data=input_data,
            expected_output_shape=(
                self.batch_size,
                self.text_sequence_length,
                8,
            ),
            variable_length_data=[input_data],
            run_quantization_check=False,
        )

    def test_embedding_model(self):
        embedding_dim = 16
        pooling_intermediate_dim = 32
        init_kwargs = self.text_init_kwargs.copy()
        input_data = self.text_backbone_input_data.copy()

        init_kwargs["is_embedding_model"] = True
        init_kwargs["embedding_dim"] = embedding_dim
        init_kwargs["pooling_intermediate_dim"] = pooling_intermediate_dim

        self.run_backbone_test(
            cls=Gemma3Backbone,
            init_kwargs=init_kwargs,
            input_data=input_data,
            expected_output_shape={
                "sequence_output": (
                    self.batch_size,
                    self.text_sequence_length,
                    8,
                ),
                "pooled_output": (self.batch_size, embedding_dim),
            },
        )

    @parameterized.named_parameters(
        ("text_and_vision", "text_and_vision", 7560, 15),
        ("text_only", "text_only", 5752, 10),
    )
    def test_architecture_characteristics(
        self, backbone_type, num_params, num_layers
    ):
        if backbone_type == "text_and_vision":
            init_kwargs = self.init_kwargs
        elif backbone_type == "text_only":
            init_kwargs = self.text_init_kwargs

        model = Gemma3Backbone(**init_kwargs)
        self.assertEqual(model.count_params(), num_params)
        self.assertEqual(len(model.layers), num_layers)

    def test_backbone_interleaved_attention(self):
        backbone = Gemma3Backbone(**self.init_kwargs)
        for i, layer in enumerate(backbone.transformer_layers):
            expected_sliding = i % 6 < 5
            self.assertEqual(
                layer.use_sliding_window_attention,
                expected_sliding,
                f"Layer {i} mismatch: expected sliding={expected_sliding}, but "
                "got {layer.use_sliding_window_attention}",
            )

    @parameterized.named_parameters(
        ("text_and_vision", "text_and_vision"), ("text_only", "text_only")
    )
    def test_saved_model(self, backbone_type):
        if backbone_type == "text_and_vision":
            init_kwargs = self.init_kwargs
            input_data = self.input_data
        elif backbone_type == "text_only":
            init_kwargs = self.text_init_kwargs
            input_data = self.text_backbone_input_data

        self.run_model_saving_test(
            cls=Gemma3Backbone,
            init_kwargs=init_kwargs,
            input_data=input_data,
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_smallest_text_preset(self):
        self.run_preset_test(
            cls=Gemma3Backbone,
            preset="gemma3_instruct_1b",
            input_data={
                "token_ids": ops.array([[651, 4320, 8426, 25341, 235265]]),
                "padding_mask": ops.ones((1, 5), dtype="int32"),
            },
            expected_output_shape=(1, 5, 1152),
            # The forward pass from a preset should be stable!
            expected_partial_output=ops.array(
                [-0.400391, -8.625, 0.605469, 1.726562, -1.507812]
            ),
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Gemma3Backbone.presets:
            self.run_preset_test(
                cls=Gemma3Backbone,
                preset=preset,
                input_data=self.text_backbone_input_data
                if "_text" in preset or "1b" in preset
                else self.input_data,
            )

    def test_distribution(self):
        # Text-only config. The default num_key_value_heads=1 is intentionally
        # left as-is (not divisible by the 2-device model-parallel mesh the
        # helper pins) to regression-test that key/value kernels are left
        # replicated on the model axis rather than sharded -- see
        # get_layout_map's comment.
        self.run_distribution_test(
            cls=Gemma3Backbone,
            init_kwargs=self.text_init_kwargs,
            input_data=self.text_backbone_input_data,
            expected_shardings={
                "token_embedding/embeddings": ("model", "batch"),
                "decoder_block.*attention/query/kernel": (
                    "model",
                    "batch",
                    None,
                ),
                "decoder_block.*attention/key/kernel": (None, "batch", None),
                "decoder_block.*attention/value/kernel": (None, "batch", None),
                "decoder_block.*attention_output/kernel": (
                    "model",
                    None,
                    "batch",
                ),
                "decoder_block.*ffw_gating/kernel": ("batch", "model"),
                "decoder_block.*ffw_gating_2/kernel": ("batch", "model"),
                "decoder_block.*ffw_linear/kernel": ("model", "batch"),
            },
            allow_replicated=(),
        )

    def test_distribution_vision(self):
        # Vision + text config -- answers the maintainer's inline comment that
        # the vision tower was previously left entirely replicated. The vision
        # encoder MUST be constructed inside the distribution scope (mixed
        # local/distributed variables otherwise crash `fit()`), so `init_kwargs`
        # is passed as a callable that the helper invokes inside the scope.
        def vision_init_kwargs():
            kwargs = copy.deepcopy(self.text_init_kwargs)
            kwargs["vision_encoder"] = Gemma3VisionEncoder(
                **self.vision_encoder_kwargs
            )
            return kwargs

        self.run_distribution_test(
            cls=Gemma3Backbone,
            init_kwargs=vision_init_kwargs,
            input_data=self.input_data,
            expected_shardings={
                # Text decoder (unchanged).
                "token_embedding/embeddings": ("model", "batch"),
                "decoder_block.*attention/query/kernel": (
                    "model",
                    "batch",
                    None,
                ),
                "decoder_block.*attention/key/kernel": (None, "batch", None),
                "decoder_block.*attention/value/kernel": (None, "batch", None),
                "decoder_block.*attention_output/kernel": (
                    "model",
                    None,
                    "batch",
                ),
                "decoder_block.*ffw_gating/kernel": ("batch", "model"),
                "decoder_block.*ffw_gating_2/kernel": ("batch", "model"),
                "decoder_block.*ffw_linear/kernel": ("model", "batch"),
                # Vision tower (newly sharded, B.2).
                "image_encoder.*multi_head_attention.*query_proj/kernel": (
                    "batch",
                    "model",
                ),
                "image_encoder.*multi_head_attention.*key_proj/kernel": (
                    "batch",
                    "model",
                ),
                "image_encoder.*multi_head_attention.*value_proj/kernel": (
                    "batch",
                    "model",
                ),
                "image_encoder.*multi_head_attention.*out_proj/kernel": (
                    "model",
                    "batch",
                ),
                "image_encoder.*mlp_dense_1/kernel": ("batch", "model"),
                "image_encoder.*mlp_dense_2/kernel": ("model", "batch"),
                "vision_output_encoder.*vision_input_projection/kernel": (
                    "model",
                    "batch",
                ),
            },
            # The patch-embedding conv (4-D) and the learned vision position
            # embedding table are intentionally left replicated -- see
            # get_layout_map's vision comment.
            allow_replicated=(
                r"image_encoder.*embedding_conv/kernel",
                r"image_encoder.*position_embedding/embeddings",
            ),
            # Vision-tower parity adds a second, larger model per mesh shape;
            # the forward/train regression on the (1, 2) mesh above already
            # exercises the numerically-sensitive vision path, so skip the
            # extra parity twins here to keep memory-constrained local
            # environments within budget.
            assert_parity_vs_undistributed=False,
        )

    @parameterized.named_parameters(
        (
            f"{dims['source_preset'].split(' ')[0]}_mesh"
            f"_{'x'.join(str(s) for s in shape)}",
            dims,
            shape,
        )
        for dims in (GEMMA3_1B_DIMS, GEMMA3_4B_DIMS, GEMMA3_27B_DIMS)
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
            # weights (no rule targets it), matching every other 3D-mesh test
            # in this PR series.
            axis_names = ("batch", "seq", "model")
        device_mesh = keras.distribution.DeviceMesh(
            shape=mesh_shape,
            axis_names=axis_names,
            devices=devices,
        )
        layout_map = Gemma3Backbone.get_layout_map(device_mesh)
        distribution = keras.distribution.ModelParallel(
            layout_map=layout_map, batch_dim_name="batch"
        )
        # Text-only sweep: the divisibility properties under test live in the
        # text decoder. `vision_encoder` is left None so this stays a small,
        # fast, spec-only build; the vision sharding is validated by
        # `test_distribution_vision` above.
        init_kwargs = {
            k: v
            for k, v in dims.items()
            if k not in ("source_preset", "image_size")
        }
        with distribution.scope():
            # bfloat16: a memory mitigation for memory-constrained local
            # environments -- spec assertions are dtype-independent.
            model = Gemma3Backbone(
                dtype="bfloat16", image_size=16, **init_kwargs
            )
            _assert_text_shardings_and_coverage(self, model, layout_map)
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
        # (e.g. base vs instruction-tuned variants of the same size, or the
        # text vs multimodal variant of one size) are only built once per mesh
        # shape -- a memory/time necessity in memory-constrained local
        # environments -- while every preset in the registry is still fetched
        # and evaluated, preserving full registry coverage.
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
        for preset in Gemma3Backbone.presets:
            try:
                path = get_file(preset, CONFIG_FILE)
                with open(path) as f:
                    cfg = json.load(f)["config"]
            except Exception as e:
                # A preset this account can't reach (e.g. an unaccepted Kaggle
                # license consent click-through) is logged, not fatal -- the
                # rest of the registry still gets exercised.
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
                "license-consent gate on this account for the gemma3 family. "
                "See the module comment above CAPPED_MESH_SHAPES."
            )
        print(
            f"test_layout_map_live_presets: {len(width_classes)} unique "
            f"width-classes across {len(Gemma3Backbone.presets)} registry "
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
                    # environments cannot build full-scale presets locally.
                    # Estimate this width-class's single-decoder-block bf16
                    # footprint (embedding table + one FFN block's 3
                    # matrices, times a 3x safety margin for JAX/XLA
                    # transient copies during construction/resharding) and
                    # skip the actual build if it exceeds a conservative
                    # local threshold. The config-fetch, dedup, and
                    # divisibility-skip logic above still exercises every
                    # registry preset either way; only the expensive
                    # build+assert step is capped.
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
                    layout_map = Gemma3Backbone.get_layout_map(device_mesh)
                    distribution = keras.distribution.ModelParallel(
                        layout_map=layout_map, batch_dim_name="batch"
                    )
                    # Use the full preset config (minus `dtype`, which may be
                    # a serialized dtype-policy dict for quantized presets and
                    # would collide with the explicit `dtype="bfloat16"`
                    # override below) rather than a dims-only allowlist --
                    # the allowlist silently dropped real architecture flags
                    # (e.g. `vision_encoder`'s nested sub-config,
                    # `query_head_dim_normalize`, rope-scaling factors), so
                    # this width-class's real preset architecture was never
                    # actually validated against `get_layout_map`'s rules.
                    init_kwargs = {k: v for k, v in cfg.items() if k != "dtype"}
                    init_kwargs["num_layers"] = 1
                    init_kwargs["image_size"] = 16
                    with distribution.scope():
                        model = Gemma3Backbone(dtype="bfloat16", **init_kwargs)
                        _assert_text_shardings_and_coverage(
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
