import copy
import gc
import json
import os
import re

import keras
import numpy as np
import pytest
from absl.testing import parameterized

from keras_hub.src.models.gemma4.gemma4_audio_encoder import Gemma4AudioEncoder
from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone
from keras_hub.src.models.gemma4.gemma4_vision_encoder import (
    Gemma4VisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.preset_utils import CONFIG_FILE
from keras_hub.src.utils.preset_utils import get_file

# Text-decoder (+ MoE expert-bank/router) expected shardings, shared by
# test_distribution and the Tier-2/Tier-3 mesh sweeps. Gemma4's attention
# kernels use the Gemma `(num_heads, hidden, head_dim)` convention (einsum
# `btd,ndh->btnh`), so heads-on-model (`query = ("model", "batch", None)`) is
# already the communication-efficient Megatron column-parallel choice and
# stays unchanged from the branch's starting point -- this PR adds the vision
# tower and audio conformer entries (see plan Part B.4), not a QKV-axis fix
# (gemma-family attention is already correct).
_TEXT_EXPECTED_SHARDINGS = {
    "token_embedding/embeddings": ("model", "batch"),
    "decoder_block.*attention/query/kernel": ("model", "batch", None),
    "decoder_block.*attention/key/kernel": (None, "batch", None),
    "decoder_block.*attention/value/kernel": (None, "batch", None),
    "decoder_block.*attention_output/kernel": ("model", None, "batch"),
    "decoder_block.*ffw_gating/kernel": ("batch", "model"),
    "decoder_block.*ffw_gating_2/kernel": ("batch", "model"),
    "decoder_block.*ffw_linear/kernel": ("model", "batch"),
    # MoE expert bank + router (26b-a4b). The expert axis (leading dim) is
    # replicated; the expert weights themselves shard hidden/intermediate.
    "decoder_block.*moe_expert_bank/gate_proj": (None, "batch", "model"),
    "decoder_block.*moe_expert_bank/up_proj": (None, "batch", "model"),
    "decoder_block.*moe_expert_bank/down_proj": (None, "model", "batch"),
    "decoder_block.*moe_router/proj/kernel": ("batch", None),
}

# Vision-tower expected shardings (plan Part B.4). The vision encoder reuses
# Gemma4 decoder blocks, so its attention/FFW kernels share the text decoder's
# 3-D Gemma-convention axis order; the clippable-einsum wrapper nests the real
# weight under `.../dense/kernel`.
_VISION_EXPECTED_SHARDINGS = {
    "image_encoder.*attention/query/dense/kernel": ("model", "batch", None),
    "image_encoder.*attention/key/dense/kernel": (None, "batch", None),
    "image_encoder.*attention/value/dense/kernel": (None, "batch", None),
    "image_encoder.*attention_output/dense/kernel": ("model", None, "batch"),
    "image_encoder.*ffw_gating/dense/kernel": ("batch", "model"),
    "image_encoder.*ffw_gating_2/dense/kernel": ("batch", "model"),
    "image_encoder.*ffw_linear/dense/kernel": ("model", "batch"),
    "vision_output_encoder.*vision_input_projection/kernel": (
        "model",
        "batch",
    ),
}

# Audio-conformer expected shardings (plan Part B.4). 2-D Dense kernels wrapped
# in the clippable-dense layer (`.../<name>_dense/kernel`); same column/row
# convention as the text/vision FFW.
_AUDIO_EXPECTED_SHARDINGS = {
    "conformer.*ffw_start_ffw_1.*dense/kernel": ("batch", "model"),
    "conformer.*ffw_start_ffw_2.*dense/kernel": ("model", "batch"),
    "conformer.*ffw_end_ffw_1.*dense/kernel": ("batch", "model"),
    "conformer.*ffw_end_ffw_2.*dense/kernel": ("model", "batch"),
    "conformer.*attention_attn_q_proj.*dense/kernel": ("batch", "model"),
    "conformer.*attention_attn_k_proj.*dense/kernel": ("batch", "model"),
    "conformer.*attention_attn_v_proj.*dense/kernel": ("batch", "model"),
    "conformer.*attention_out_proj.*dense/kernel": ("model", "batch"),
    "conformer.*lconv_linear_start.*dense/kernel": ("batch", "model"),
    "conformer.*lconv_linear_end.*dense/kernel": ("model", "batch"),
    "output_proj/kernel": ("model", "batch"),
    "audio_output_projection/kernel": ("batch", "model"),
}

# Intentionally-replicated rank>=2 weights (plan Part B.4). Exhaustive: every
# rank>=2 weight not covered by a layout-map rule must match one of these.
_VISION_ALLOW_REPLICATED = (
    r"patch_embedder/input_proj/kernel",
    r"patch_embedder/position_embedding_table",
)
_AUDIO_ALLOW_REPLICATED = (
    r"(^|/)conv/kernel",  # sub-sampling Conv2D stack
    r"depthwise_conv/kernel",  # conformer depthwise Conv1D
    r"rpe/pos_proj",  # relative-position projection
    r"(^|/)input_proj/kernel",  # mel-feature -> hidden input projection
)
# Per-layer-input weights (built only when `hidden_size_per_layer_input > 0`,
# see `get_layout_map`'s docstring). Covers both the backbone-level weights
# (`per_layer_token_embedding`, `per_layer_model_projection`) and the
# per-decoder-block weights (`per_layer_input_gate`, `per_layer_up_proj` --
# see `Gemma4TextDecoderBlock.__init__`), which are gated by the same flag
# and are equally unmapped. No current registry preset enables this feature,
# but the Tier-2/Tier-3 mesh sweeps must not fail coverage the moment a
# future preset or width class does.
_TEXT_ALLOW_REPLICATED = (
    r"per_layer_token_embedding/embeddings",
    r"per_layer_model_projection/kernel",
    r"decoder_block.*per_layer_input_gate/kernel",
    r"decoder_block.*per_layer_up_proj/kernel",
)

# Dims for the Tier-2 CI-safe mesh-shape sweep: representative real-preset
# dimensions, frozen as literals and sourced once, offline -- do not add a
# `get_file` call to the Tier-2 test body itself (that is what Tier 3,
# `test_layout_map_live_presets` below, is for).
#
# MEMORY NOTE: memory-constrained local environments cannot load full-scale
# model dims. What actually matters for the divisibility/sharding properties
# this tier tests is the RATIO of query heads to kv heads and whether
# hidden/intermediate/vocab divide the mesh's model-axis sizes -- not the
# absolute parameter count. So
# these dims are scaled down by roughly 20-30x from the real gemma4 presets
# while preserving each preset's real query:kv head ratio (gemma4 E2B/E4B GQA
# and the 26b-a4b MoE) and keeping hidden/intermediate/vocab as clean multiples
# divisible by every mesh shape in CAPPED_MESH_SHAPES. Full-scale real dims are
# exercised by `test_layout_map_live_presets` below, which has its own
# per-width-class memory-budget skip so it never attempts a full-scale build
# locally either -- true full-scale verification happens offline on a machine
# with more RAM.
GEMMA4_2B_DIMS = {
    "source_preset": "gemma4_2b (real ratio, memory-scaled dims)",
    "vocabulary_size": 2048,
    "image_size": 16,
    "num_layers": 1,  # depth is irrelevant to spec matching/divisibility.
    "num_query_heads": 8,
    "num_key_value_heads": 2,  # real ratio: GQA, 4:1.
    "hidden_dim": 256,
    "intermediate_dim": 1024,
    "head_dim": 32,
}
GEMMA4_4B_DIMS = {
    "source_preset": "gemma4_4b (real ratio, memory-scaled dims)",
    "vocabulary_size": 2048,
    "image_size": 16,
    "num_layers": 1,
    "num_query_heads": 8,
    "num_key_value_heads": 4,  # real ratio: GQA, 2:1.
    "hidden_dim": 384,
    "intermediate_dim": 1536,
    "head_dim": 32,
}
# The 26b-a4b Mixture-of-Experts width class. Includes the parallel MoE block
# so the Tier-2 sweep also exercises the expert-bank/router layout rules across
# every mesh shape.
GEMMA4_26B_A4B_DIMS = {
    "source_preset": "gemma4_26b_a4b (real ratio, memory-scaled dims)",
    "vocabulary_size": 2048,
    "image_size": 16,
    "num_layers": 1,
    "num_query_heads": 16,
    "num_key_value_heads": 8,  # real ratio: GQA, 2:1.
    "hidden_dim": 512,
    "intermediate_dim": 2048,
    "head_dim": 32,
    "enable_moe_block": True,
    "num_experts": 8,
    "expert_intermediate_dim": 256,
    "num_experts_per_token": 2,
}

# Hard-capped mesh-shape list for memory-constrained local environments. The
# full 10-shape matrix from the testing-strategy doc is
# 2x4, 1x8, 4x4, 8x8, 16x16, 2x2x2, 1x1x8, 2x2x4, 4x4x4, 4x4x8 -- shapes
# 8x8, 16x16, 4x4x4, 4x4x8 (64-256 virtual devices) are DELIBERATELY DROPPED
# here: they exceed a memory-constrained local environment's memory budget (a
# prior attempt at this pipeline was OOM-killed). These shapes require a
# dedicated or CI machine with more memory -- do not attempt them
# experimentally in a memory-constrained local environment.
CAPPED_MESH_SHAPES = [
    (2, 4),
    (1, 8),
    (4, 4),
    (2, 2, 2),
    (1, 1, 8),
    (2, 2, 4),
]


def _assert_text_shardings_and_coverage(
    test_case, model, layout_map, allow_replicated=_TEXT_ALLOW_REPLICATED
):
    """Shared spec + coverage assertions for the Tier-2/3 text mesh sweeps."""
    for pattern, expected in _TEXT_EXPECTED_SHARDINGS.items():
        matches = [w for w in model.weights if re.search(pattern, w.path)]
        # MoE rules only match when the width class enables the MoE block; a
        # dense width class legitimately has zero matches for them.
        if "moe_" in pattern and not matches:
            continue
        test_case.assertGreater(len(matches), 0)
        for w in matches:
            test_case.assertEqual(tuple(w.value.sharding.spec), expected)
    offending = [
        w.path
        for w in model.weights
        if len(w.shape) >= 2
        and layout_map[w.path] is None
        and not any(re.search(p, w.path) for p in allow_replicated)
    ]
    test_case.assertEqual(
        offending,
        [],
        f"The following rank>=2 weights are unmapped: {offending}",
    )


class Gemma4BackboneTest(TestCase, parameterized.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.vocabulary_size = 256
        self.text_sequence_length = 64
        self.image_size = 16
        # (image_size / patch_size)^2 / pool_size^2 = (16/4)^2 / 2^2 = 4
        self.vision_tokens_per_image = int((self.image_size / 4) ** 2 // 4)
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
            "head_dim": 4,
            "num_key_value_heads": 2,
            "hidden_dim": 8,
            "intermediate_dim": 16,
            "output_dim": 8,
        }

        # === Vision + Text Backbone ===
        vision_encoder = Gemma4VisionEncoder(**self.vision_encoder_kwargs)

        self.init_kwargs = {
            "vocabulary_size": self.vocabulary_size,
            "image_size": self.image_size,
            "num_layers": 6,
            "num_query_heads": 2,
            "num_key_value_heads": 1,
            "hidden_dim": 8,
            "intermediate_dim": 16,
            "head_dim": 4,
            "use_sliding_window_attention": True,
            "sliding_window_size": 16,
            "attention_logit_soft_cap": None,
            "final_logit_soft_cap": None,
            "vision_encoder": vision_encoder,
        }

        num_patches = int((self.image_size / 4) ** 2)
        patch_dim = 3 * 4 * 4
        dummy_pixel_values = np.random.rand(
            self.batch_size,
            self.max_images_per_prompt,
            num_patches,
            patch_dim,
        ).astype("float32")
        dummy_pixel_pos = np.ones(
            (self.batch_size, self.max_images_per_prompt, num_patches, 2),
            dtype="int32",
        )
        dummy_text_token_ids = np.random.rand(
            self.batch_size, self.text_sequence_length
        )

        self.input_data = {
            "token_ids": dummy_text_token_ids,
            "pixel_values": dummy_pixel_values,
            "pixel_position_ids": dummy_pixel_pos,
            "padding_mask": np.ones(
                (self.batch_size, self.text_sequence_length),
                dtype="int32",
            ),
            "position_ids": np.tile(
                np.arange(self.text_sequence_length, dtype="int32")[
                    np.newaxis, :
                ],
                (self.batch_size, 1),
            ),
        }
        # 4 vision tokens per image; 3 images per sample => 12 vision tokens
        vision_mask_0 = (
            [False] * 20
            + [True] * 4
            + [False] * 16
            + [True] * 4
            + [False] * 16
            + [True] * 4
        )
        vision_mask_1 = (
            [False] * 16
            + [True] * 4
            + [False] * 16
            + [True] * 4
            + [False] * 20
            + [True] * 4
        )
        self.input_data["vision_mask"] = np.array(
            [vision_mask_0, vision_mask_1]
        )
        self.input_data["vision_indices"] = np.array(
            [
                list(range(20, 24)) + list(range(40, 44)) + list(range(60, 64)),
                list(range(16, 20)) + list(range(36, 40)) + list(range(60, 64)),
            ]
        )

        # === Text-only Backbone ===
        self.text_init_kwargs = copy.deepcopy(self.init_kwargs)
        del self.text_init_kwargs["vision_encoder"]

        self.text_backbone_input_data = copy.deepcopy(self.input_data)
        del self.text_backbone_input_data["pixel_values"]
        del self.text_backbone_input_data["pixel_position_ids"]
        del self.text_backbone_input_data["vision_mask"]
        del self.text_backbone_input_data["vision_indices"]

        # === MoE text config (exercises the expert-bank/router layout) ===
        self.moe_init_kwargs = copy.deepcopy(self.text_init_kwargs)
        self.moe_init_kwargs["enable_moe_block"] = True
        self.moe_init_kwargs["num_experts"] = 4
        self.moe_init_kwargs["expert_intermediate_dim"] = 8
        self.moe_init_kwargs["num_experts_per_token"] = 2

        # === Audio encoder config (tiny, matches test_audio_backbone_basics,
        # except `output_proj_dims` -- see below) ==
        self.audio_input_feat_size = 8
        self.audio_num_tokens_per_clip = 4
        self.audio_encoder_kwargs = {
            "input_feat_size": self.audio_input_feat_size,
            "hidden_size": 8,
            "num_heads": 2,
            "num_layers": 1,
            "chunk_size": 4,
            "context_left": 5,
            "context_right": 0,
            "sscp_conv_channels": (4, 2),
            "sscp_kernel_sizes": ((3, 3), (3, 3)),
            "sscp_stride_sizes": ((2, 2), (2, 2)),
            # Non-`None` (unlike test_audio_backbone_basics' config): the
            # class default is `1536`, so real presets build the
            # `output_proj` weight. Set here (not `None`) so
            # test_distribution_audio actually builds and exercises the
            # `output_proj/kernel` layout-map rule / coverage assertion
            # instead of silently avoiding it.
            "output_proj_dims": 6,
            "output_dim": 8,
        }
        num_clips = 1
        audio_T = 16
        self.audio_input_data = {
            **self.text_backbone_input_data,
            "audio_mel": np.random.rand(
                self.batch_size, num_clips, audio_T, self.audio_input_feat_size
            ).astype("float32"),
            "audio_mel_mask": np.ones(
                (self.batch_size, num_clips, audio_T), dtype="int32"
            ),
            "audio_indices": np.zeros(
                (
                    self.batch_size,
                    num_clips * self.audio_num_tokens_per_clip,
                ),
                dtype="int32",
            ),
            "audio_mask": np.zeros(
                (self.batch_size, self.text_sequence_length),
                dtype="int32",
            ),
        }

    def test_audio_backbone_basics(self):
        """Backbone with audio encoder."""

        input_feat_size = 8
        num_audio_tokens_per_clip = 4
        audio_encoder = Gemma4AudioEncoder(
            input_feat_size=input_feat_size,
            hidden_size=8,
            num_heads=2,
            num_layers=1,
            chunk_size=4,
            context_left=5,
            context_right=0,
            sscp_conv_channels=(4, 2),
            sscp_kernel_sizes=((3, 3), (3, 3)),
            sscp_stride_sizes=((2, 2), (2, 2)),
            output_proj_dims=None,
            output_dim=8,
        )
        audio_init_kwargs = {
            **self.text_init_kwargs,
            "audio_encoder": audio_encoder,
            "num_audio_tokens_per_clip": num_audio_tokens_per_clip,
        }
        # Text + audio input: N_clips=1, T=16, F=input_feat_size.
        num_clips = 1
        T = 16
        audio_input_data = {
            **self.text_backbone_input_data,
            "audio_mel": np.random.rand(
                self.batch_size, num_clips, T, input_feat_size
            ).astype("float32"),
            "audio_mel_mask": np.ones(
                (self.batch_size, num_clips, T), dtype="int32"
            ),
            "audio_indices": np.zeros(
                (self.batch_size, num_clips * num_audio_tokens_per_clip),
                dtype="int32",
            ),
            "audio_mask": np.zeros(
                (self.batch_size, self.text_sequence_length),
                dtype="int32",
            ),
        }
        backbone = Gemma4Backbone(**audio_init_kwargs)
        output = backbone(audio_input_data)
        self.assertEqual(
            output.shape,
            (self.batch_size, self.text_sequence_length, 8),
        )

        # Also test with N_clips=0 (the dummy / text-only path used by the
        # preprocessor when no audio is present in the prompt).
        dummy_audio_data = {
            **self.text_backbone_input_data,
            "audio_mel": np.zeros(
                (self.batch_size, 0, 1, input_feat_size), dtype="float32"
            ),
            "audio_mel_mask": np.zeros((self.batch_size, 0, 1), dtype="int32"),
            "audio_indices": np.zeros((self.batch_size, 0), dtype="int32"),
            "audio_mask": np.zeros(
                (self.batch_size, self.text_sequence_length),
                dtype="int32",
            ),
        }
        output_dummy = backbone(dummy_audio_data)
        self.assertEqual(
            output_dummy.shape,
            (self.batch_size, self.text_sequence_length, 8),
        )

    @parameterized.named_parameters(
        ("text_and_vision", "text_and_vision"), ("text_only", "text_only")
    )
    def test_backbone_basics(self, backbone_type):
        if backbone_type == "text_and_vision":
            init_kwargs = self.init_kwargs
            input_data = self.input_data
        else:
            init_kwargs = self.text_init_kwargs
            input_data = self.text_backbone_input_data

        self.run_backbone_test(
            cls=Gemma4Backbone,
            init_kwargs=init_kwargs,
            input_data=input_data,
            expected_output_shape=(
                self.batch_size,
                self.text_sequence_length,
                8,
            ),
            variable_length_data=[input_data],
            run_quantization_check=False,
            # The vision encoder is intentionally float32-only, so it does not
            # follow the mixed-precision policy applied to the backbone.
            run_mixed_precision_check=(backbone_type != "text_and_vision"),
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
            cls=Gemma4Backbone,
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
        ("text_and_vision", "text_and_vision", 24006, 17),
        ("text_only", "text_only", 5758, 11),
    )
    def test_architecture_characteristics(
        self, backbone_type, num_params, num_layers
    ):
        if backbone_type == "text_and_vision":
            init_kwargs = self.init_kwargs
        else:
            init_kwargs = self.text_init_kwargs

        model = Gemma4Backbone(**init_kwargs)
        self.assertEqual(model.count_params(), num_params)
        self.assertEqual(len(model.layers), num_layers)

    def test_backbone_layer_attention_pattern(self):
        """Verifies every 6th layer (0-indexed 5, 11, ...) is global."""
        backbone = Gemma4Backbone(**self.text_init_kwargs)
        for i, layer in enumerate(backbone.transformer_layers):
            expected_global = (i % 6) == 5
            expected_sliding = (
                not expected_global
            ) and backbone.use_sliding_window_attention
            self.assertEqual(
                layer.use_sliding_window_attention,
                expected_sliding,
                f"Layer {i}: expected sliding={expected_sliding}, got "
                f"{layer.use_sliding_window_attention}",
            )

    def test_all_text_layers_have_layer_scalar(self):
        """All text decoder layers should expose a layer_scalar weight."""
        backbone = Gemma4Backbone(**self.text_init_kwargs)
        for i, layer in enumerate(backbone.transformer_layers):
            self.assertTrue(
                hasattr(layer, "layer_scalar"),
                f"Text decoder layer {i} missing layer_scalar",
            )

    def test_moe_architecture(self):
        """MoE blocks (parallel dense + expert paths) should run end-to-end."""
        model = Gemma4Backbone(**self.moe_init_kwargs)
        output = model(self.text_backbone_input_data)
        self.assertEqual(
            output.shape,
            (self.batch_size, self.text_sequence_length, 8),
        )

    def test_partial_rotary(self):
        """Partial RoPE (global_rope_partial_rotary_factor < 1) should run."""
        init_kwargs = copy.deepcopy(self.text_init_kwargs)
        init_kwargs["global_rope_partial_rotary_factor"] = 0.25
        model = Gemma4Backbone(**init_kwargs)
        output = model(self.text_backbone_input_data)
        self.assertEqual(
            output.shape,
            (self.batch_size, self.text_sequence_length, 8),
        )

    def test_double_wide_mlp(self):
        """KV-shared layers should use 2× intermediate_dim when requested."""
        init_kwargs = copy.deepcopy(self.text_init_kwargs)
        init_kwargs["use_double_wide_mlp"] = True
        init_kwargs["num_kv_shared_layers"] = 3
        model = Gemma4Backbone(**init_kwargs)
        output = model(self.text_backbone_input_data)
        self.assertEqual(
            output.shape,
            (self.batch_size, self.text_sequence_length, 8),
        )

    @parameterized.named_parameters(
        ("text_and_vision", "text_and_vision"), ("text_only", "text_only")
    )
    def test_saved_model(self, backbone_type):
        if backbone_type == "text_and_vision":
            init_kwargs = self.init_kwargs
            input_data = self.input_data
        else:
            init_kwargs = self.text_init_kwargs
            input_data = self.text_backbone_input_data

        self.run_model_saving_test(
            cls=Gemma4Backbone,
            init_kwargs=init_kwargs,
            input_data=input_data,
        )

    def test_distribution(self):
        # Text-only config with the MoE block enabled so the expert-bank/router
        # layout rules are also exercised. The default num_key_value_heads=1 is
        # intentionally left as-is (not divisible by the 2-device
        # model-parallel mesh the helper pins) to regression-test that
        # key/value kernels are left replicated on the model axis rather than
        # sharded -- see get_layout_map's comment: `effective_num_kv_heads` can
        # be small and independent of num_query_heads (e.g. global-attention
        # layers via num_global_key_value_heads), so sharding it on the model
        # axis raises an IndivisibleError whenever the device count doesn't
        # divide that head count.
        self.run_distribution_test(
            cls=Gemma4Backbone,
            init_kwargs=self.moe_init_kwargs,
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
                "decoder_block.*moe_expert_bank/gate_proj": (
                    None,
                    "batch",
                    "model",
                ),
                "decoder_block.*moe_expert_bank/up_proj": (
                    None,
                    "batch",
                    "model",
                ),
                "decoder_block.*moe_expert_bank/down_proj": (
                    None,
                    "model",
                    "batch",
                ),
                "decoder_block.*moe_router/proj/kernel": ("batch", None),
            },
            allow_replicated=(),
            # MoE model: use the looser MoE parity tolerances (sharded
            # top-k-reduction float noise is larger than dense) -- see the
            # helper docstring / plan Section C.1.
            is_moe=True,
        )

    def test_distribution_vision(self):
        # Vision + text config -- validates the vision-tower sharding entries
        # added in this PR (plan Part B.4). The vision encoder MUST be
        # constructed inside the distribution scope (mixed local/distributed
        # variables otherwise crash `fit()`), so `init_kwargs` is passed as a
        # callable that the helper invokes inside the scope.
        def vision_init_kwargs():
            kwargs = copy.deepcopy(self.text_init_kwargs)
            kwargs["vision_encoder"] = Gemma4VisionEncoder(
                **self.vision_encoder_kwargs
            )
            return kwargs

        self.run_distribution_test(
            cls=Gemma4Backbone,
            init_kwargs=vision_init_kwargs,
            input_data=self.input_data,
            expected_shardings={
                # Text decoder (unchanged).
                **{
                    k: v
                    for k, v in _TEXT_EXPECTED_SHARDINGS.items()
                    if "moe_" not in k
                },
                # Vision tower (newly sharded, B.4).
                **_VISION_EXPECTED_SHARDINGS,
            },
            allow_replicated=_VISION_ALLOW_REPLICATED,
            # Vision-tower parity adds a second, larger model per mesh shape;
            # the forward/train regression on the (1, 2) mesh above already
            # exercises the numerically-sensitive vision path, so skip the
            # extra parity twins here to keep memory-constrained local
            # environments within budget.
            assert_parity_vs_undistributed=False,
        )

    def test_distribution_audio(self):
        # Text + audio config -- validates the audio-conformer sharding entries
        # added in this PR (plan Part B.4). Forward-only (no fit, no parity
        # twins): the conformer is heavily convolutional / non-standard
        # attention and memory-constrained local environments are RAM-limited,
        # so this test just asserts the spec + coverage nets and a finite
        # forward pass. The audio encoder is float32-only, so build it inside
        # the scope via a callable.
        def audio_init_kwargs():
            kwargs = copy.deepcopy(self.text_init_kwargs)
            kwargs["audio_encoder"] = Gemma4AudioEncoder(
                **self.audio_encoder_kwargs
            )
            kwargs["num_audio_tokens_per_clip"] = self.audio_num_tokens_per_clip
            return kwargs

        self.run_distribution_test(
            cls=Gemma4Backbone,
            init_kwargs=audio_init_kwargs,
            input_data=self.audio_input_data,
            expected_shardings={
                # Text decoder (unchanged).
                **{
                    k: v
                    for k, v in _TEXT_EXPECTED_SHARDINGS.items()
                    if "moe_" not in k
                },
                # Audio conformer (newly sharded, B.4).
                **_AUDIO_EXPECTED_SHARDINGS,
            },
            allow_replicated=_AUDIO_ALLOW_REPLICATED,
            run_training=False,
            assert_parity_vs_undistributed=False,
        )

    @parameterized.named_parameters(
        (
            f"{dims['source_preset'].split(' ')[0]}_mesh"
            f"_{'x'.join(str(s) for s in shape)}",
            dims,
            shape,
        )
        for dims in (GEMMA4_2B_DIMS, GEMMA4_4B_DIMS, GEMMA4_26B_A4B_DIMS)
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
        layout_map = Gemma4Backbone.get_layout_map(device_mesh)
        distribution = keras.distribution.ModelParallel(
            layout_map=layout_map, batch_dim_name="batch"
        )
        # Text-only sweep (vision_encoder / audio_encoder left None): the
        # divisibility properties under test live in the text decoder and the
        # MoE expert bank. The vision and audio sharding is validated by
        # test_distribution_vision / test_distribution_audio above.
        init_kwargs = {
            k: v
            for k, v in dims.items()
            if k not in ("source_preset", "image_size")
        }
        with distribution.scope():
            # bfloat16: a memory mitigation for memory-constrained local
            # environments -- spec assertions are dtype-independent.
            model = Gemma4Backbone(
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
            "head_dim",
        )
        width_classes = {}  # dedupe key -> (config dict, [preset names])
        fetch_failures = []
        for preset in Gemma4Backbone.presets:
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
                "license-consent gate on this account for the gemma4 family, "
                "or missing Kaggle credentials. See the module comment above "
                "CAPPED_MESH_SHAPES."
            )
        print(
            f"test_layout_map_live_presets: {len(width_classes)} unique "
            f"width-classes across {len(Gemma4Backbone.presets)} registry "
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
                    # footprint (embedding table + one FFN block's 3
                    # matrices, times a 3x safety margin for JAX/XLA
                    # transient copies during construction/resharding) and
                    # skip the actual build if it exceeds a conservative
                    # local threshold. The config-fetch, dedup, and
                    # divisibility-skip logic above still exercises every
                    # registry preset either way; only the expensive
                    # build+assert step is capped.
                    #
                    # MoE presets (26b-a4b) additionally include an
                    # expert-bank term: gate_proj + up_proj (hidden ->
                    # expert_intermediate_dim each) + down_proj
                    # (expert_intermediate_dim -> hidden), replicated across
                    # num_experts. Omitting this term would dangerously
                    # undercount a MoE width-class's real footprint, the
                    # same way it would for gpt_oss's expert banks (see
                    # gpt_oss_backbone_test.py's identical guard).
                    est_params = (
                        cfg["vocabulary_size"] * cfg["hidden_dim"]
                        + 3 * cfg["hidden_dim"] * cfg["intermediate_dim"]
                    )
                    if cfg.get("enable_moe_block"):
                        est_params += cfg["num_experts"] * (
                            2
                            * cfg["hidden_dim"]
                            * cfg["expert_intermediate_dim"]
                            + cfg["expert_intermediate_dim"] * cfg["hidden_dim"]
                        )
                    est_bytes = est_params * 2 * 3  # bf16 * safety margin
                    # Tunable via env var so CI or a bigger machine can opt
                    # into real full-scale verification; defaults to 300MB
                    # to preserve today's behavior in memory-constrained
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
                    layout_map = Gemma4Backbone.get_layout_map(device_mesh)
                    distribution = keras.distribution.ModelParallel(
                        layout_map=layout_map, batch_dim_name="batch"
                    )
                    # Use the full preset config (minus `dtype`, which may be
                    # a serialized dtype-policy dict for quantized presets
                    # and would collide with the explicit `dtype="bfloat16"`
                    # override below) rather than a dims-only allowlist --
                    # the allowlist silently dropped real architecture flags
                    # (`enable_moe_block`/`num_experts`/
                    # `expert_intermediate_dim`, `layer_types`,
                    # `global_head_dim`/`num_global_key_value_heads`, and the
                    # entire vision/audio sub-configs), so this width-class's
                    # real preset architecture was never actually validated
                    # against `get_layout_map`'s rules.
                    init_kwargs = {k: v for k, v in cfg.items() if k != "dtype"}
                    init_kwargs["num_layers"] = 1
                    init_kwargs["image_size"] = 16
                    # Keep `layer_types`/`num_kv_shared_layers` consistent
                    # with the forced single layer above: a real preset's
                    # `layer_types` list (length == its real num_layers) or
                    # a nonzero `num_kv_shared_layers` would otherwise be
                    # combined with `num_layers=1`, which can drive
                    # `_first_kv_shared` negative inside `__init__` and
                    # silently build an inconsistent KV-sharing map rather
                    # than raising -- mirrors t5gemma_backbone_test.py's
                    # identical `encoder_layer_types`/`decoder_layer_types`
                    # override in this same test.
                    init_kwargs["layer_types"] = ["full_attention"]
                    init_kwargs["num_kv_shared_layers"] = 0
                    # `vision_encoder`/`audio_encoder` are serialized as
                    # nested Keras-layer config dicts in the raw preset
                    # config (see `Gemma4Backbone.get_config`); `__init__`
                    # requires live layer instances (it calls e.g.
                    # `vision_encoder.num_vision_tokens_per_image`), so
                    # deserialize them the same way
                    # `Gemma4Backbone.from_config` does before constructing
                    # directly. Nearly every real gemma4 preset is
                    # audio+vision+text or vision+text (see
                    # gemma4_presets.py), so this path is exercised by most
                    # width-classes, not a rare edge case.
                    if init_kwargs.get("vision_encoder") is not None:
                        init_kwargs["vision_encoder"] = (
                            keras.layers.deserialize(
                                init_kwargs["vision_encoder"]
                            )
                        )
                    if init_kwargs.get("audio_encoder") is not None:
                        init_kwargs["audio_encoder"] = keras.layers.deserialize(
                            init_kwargs["audio_encoder"]
                        )
                    with distribution.scope():
                        model = Gemma4Backbone(dtype="bfloat16", **init_kwargs)
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

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Gemma4Backbone.presets:
            self.run_preset_test(
                cls=Gemma4Backbone,
                preset=preset,
                input_data=self.text_backbone_input_data,
            )
