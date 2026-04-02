import copy

import numpy as np
import pytest
from absl.testing import parameterized

from keras_hub.src.models.gemma4.gemma4_audio_encoder import Gemma4AudioEncoder
from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone
from keras_hub.src.models.gemma4.gemma4_vision_encoder import (
    Gemma4VisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase


class Gemma4BackboneTest(TestCase, parameterized.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.vocabulary_size = 256
        self.text_sequence_length = 64
        self.image_size = 16
        # (image_size / patch_size)^2 / pool_size^2 = (16/4)^2 / 2^2 = 4
        self.vision_tokens_per_image = int((self.image_size / 4) ** 2 // 4)
        self.max_images_per_prompt = 3

        # === Vision + Text Backbone ===
        vision_encoder = Gemma4VisionEncoder(
            image_size=self.image_size,
            patch_size=4,
            pool_size=2,
            num_layers=2,
            num_heads=2,
            head_dim=4,
            num_key_value_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            output_dim=8,
        )

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
        ("text_and_vision", "text_and_vision", 24006, 16),
        ("text_only", "text_only", 5758, 10),
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
        init_kwargs = copy.deepcopy(self.text_init_kwargs)
        init_kwargs["enable_moe_block"] = True
        init_kwargs["num_experts"] = 4
        init_kwargs["expert_intermediate_dim"] = 8
        init_kwargs["num_experts_per_token"] = 2
        model = Gemma4Backbone(**init_kwargs)
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

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Gemma4Backbone.presets:
            self.run_preset_test(
                cls=Gemma4Backbone,
                preset=preset,
                input_data=self.text_backbone_input_data,
            )
