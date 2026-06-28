import copy

import numpy as np
import pytest
from absl.testing import parameterized

from keras_hub.src.models.gemma4.gemma4_audio_encoder import Gemma4AudioEncoder
from keras_hub.src.models.gemma4_unified.gemma4_unified_backbone import (
    Gemma4UnifiedBackbone,
)
from keras_hub.src.models.gemma4_unified.gemma4_unified_vision_embedder import (
    Gemma4UnifiedVisionEmbedder,
)
from keras_hub.src.tests.test_case import TestCase


class Gemma4UnifiedBackboneTest(TestCase, parameterized.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.vocabulary_size = 256
        self.text_sequence_length = 64
        self.num_vision_tokens_per_image = 4

        # === Vision + Text Backbone (Unified / encoder-free) ===
        vision_embedder = Gemma4UnifiedVisionEmbedder(
            hidden_dim=8,
            model_patch_size=4,
            mm_posemb_size=4,
            num_soft_tokens=self.num_vision_tokens_per_image,
        )

        self.init_kwargs = {
            "vocabulary_size": self.vocabulary_size,
            "image_size": None,
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
            "vision_encoder": vision_embedder,
        }

        # Pixel values: (B, n_images, n_patches, patch_dim)
        patch_dim = 3 * 4 * 4  # model_patch_size=4
        dummy_pixel_values = np.random.rand(
            self.batch_size,
            2,
            self.num_vision_tokens_per_image,
            patch_dim,
        ).astype("float32")
        dummy_pixel_pos = np.ones(
            (
                self.batch_size,
                2,
                self.num_vision_tokens_per_image,
                2,
            ),
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
        # 4 vision tokens per image × 2 images = 8 vision tokens
        vision_mask_0 = (
            [False] * 20 + [True] * 4 + [False] * 16 + [True] * 4 + [False] * 20
        )
        vision_mask_1 = (
            [False] * 16 + [True] * 4 + [False] * 16 + [True] * 4 + [False] * 24
        )
        self.input_data["vision_mask"] = np.array(
            [vision_mask_0, vision_mask_1]
        )
        self.input_data["vision_indices"] = np.array(
            [
                list(range(20, 24)) + list(range(40, 44)),
                list(range(16, 20)) + list(range(36, 40)),
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
        backbone = Gemma4UnifiedBackbone(**audio_init_kwargs)
        output = backbone(audio_input_data)
        self.assertEqual(
            output.shape,
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
            cls=Gemma4UnifiedBackbone,
            init_kwargs=init_kwargs,
            input_data=input_data,
            expected_output_shape=(
                self.batch_size,
                self.text_sequence_length,
                8,
            ),
            variable_length_data=[input_data],
            run_quantization_check=False,
            run_mixed_precision_check=(backbone_type != "text_and_vision"),
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
            cls=Gemma4UnifiedBackbone,
            init_kwargs=init_kwargs,
            input_data=input_data,
        )

    def test_moe_architecture(self):
        """MoE blocks should run end-to-end."""
        init_kwargs = copy.deepcopy(self.text_init_kwargs)
        init_kwargs["enable_moe_block"] = True
        init_kwargs["num_experts"] = 4
        init_kwargs["expert_intermediate_dim"] = 8
        init_kwargs["num_experts_per_token"] = 2
        model = Gemma4UnifiedBackbone(**init_kwargs)
        output = model(self.text_backbone_input_data)
        self.assertEqual(
            output.shape,
            (self.batch_size, self.text_sequence_length, 8),
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Gemma4UnifiedBackbone.presets:
            self.run_preset_test(
                cls=Gemma4UnifiedBackbone,
                preset=preset,
                input_data=self.text_backbone_input_data,
            )
