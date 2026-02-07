"""Tests for Qwen3-Omni backbone."""

import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.qwen3_omni.qwen3_omni_backbone import (
    Qwen3OmniBackbone,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen3OmniBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 20,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "hidden_dim": 16,
            "intermediate_dim": 32,
            "head_dim": 4,
            "moe_intermediate_dim": 16,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "norm_topk_prob": True,
            "decoder_sparse_step": 1,
            "mrope_section": (1, 1, 0),
            "layer_norm_epsilon": 1e-6,
            "rope_max_wavelength": 10000,
            "rope_scaling_factor": 1.0,
            "dropout": 0.0,
            "tie_word_embeddings": False,
            "sliding_window_size": None,
            "router_aux_loss_coefficient": 0.001,
            "mlp_only_layers": [],
            "dtype": "float32",
        }
        self.input_data = {
            "token_ids": ops.ones((2, 8), dtype="int32"),
            "padding_mask": ops.ones((2, 8), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=Qwen3OmniBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 8, 16),
            run_quantization_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Qwen3OmniBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_architecture_characteristics(self):
        model = Qwen3OmniBackbone(**self.init_kwargs)
        self.assertEqual(len(model.transformer_layers), 2)
        self.assertGreater(model.count_params(), 0)

    def test_auxiliary_loss(self):
        model = Qwen3OmniBackbone(**self.init_kwargs)
        _ = model(self.input_data, training=True)
        self.assertTrue(
            len(model.losses) > 0, "Auxiliary losses should be present"
        )
        for loss in model.losses:
            self.assertGreater(loss, 0.0, "Auxiliary loss should be positive")

    def test_vision_encoder_integration(self):
        """Attach a tiny vision encoder and verify it is stored."""
        from keras_hub.src.models.qwen3_omni.qwen3_omni_vision_encoder import (
            Qwen3OmniVisionEncoder,
        )

        vision_encoder = Qwen3OmniVisionEncoder(
            depth=2,
            hidden_size=32,
            num_heads=4,
            intermediate_size=64,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=16,
            num_position_embeddings=49,
            deepstack_visual_indexes=[0],
            dtype="float32",
        )
        init_kwargs = self.init_kwargs.copy()
        init_kwargs["vision_encoder"] = vision_encoder
        model = Qwen3OmniBackbone(**init_kwargs)

        self.assertIsNotNone(model.vision_encoder)
        self.assertIsNone(model.audio_encoder)
        self.assertEqual(model.vision_encoder.hidden_size, 32)
        self.assertEqual(model.vision_encoder.depth, 2)
        self.assertEqual(model.vision_encoder.out_hidden_size, 16)

    def test_vision_fusion_forward(self):
        """End-to-end forward pass with vision fusion.

        Verifies that image token positions are actually replaced
        by the vision encoder output (not just that shape is correct).
        """
        from keras_hub.src.models.qwen3_omni.qwen3_omni_vision_encoder import (
            Qwen3OmniVisionEncoder,
        )

        vision_encoder = Qwen3OmniVisionEncoder(
            depth=2,
            hidden_size=32,
            num_heads=4,
            intermediate_size=64,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=16,
            num_position_embeddings=49,
            deepstack_visual_indexes=[0],
            dtype="float32",
        )
        init_kwargs = self.init_kwargs.copy()
        init_kwargs["vision_encoder"] = vision_encoder
        # Use small token IDs that fit within vocabulary_size=20.
        init_kwargs["image_token_id"] = 5
        init_kwargs["video_token_id"] = 6
        model = Qwen3OmniBackbone(**init_kwargs)

        # Build token sequence: batch=1, seq=8.
        # Place one image token (id=5) at position 3.
        token_ids = np.ones((1, 8), dtype="int32")
        token_ids[0, 3] = 5
        padding_mask = np.ones((1, 8), dtype="int32")

        # Pixel values: 1 image, 1 temporal frame.
        # Full spatial = grid_h * patch_size = 2*2 = 4.
        # With temporal_patch_size=2, T_full = 1*2 = 2.
        pixel_values = (
            np.random.RandomState(0).randn(1, 2, 4, 4, 3).astype("float32")
        )
        grid_thw = np.array([[1, 2, 2]], dtype="int32")

        output_fused = model(
            {
                "token_ids": token_ids,
                "padding_mask": padding_mask,
                "pixel_values": pixel_values,
                "grid_thw": grid_thw,
            }
        )
        self.assertEqual(ops.shape(output_fused), (1, 8, 16))

        # Compare with text-only (no pixel_values).
        output_text = model(
            {
                "token_ids": token_ids,
                "padding_mask": padding_mask,
            }
        )
        self.assertEqual(ops.shape(output_text), (1, 8, 16))

        # Fused output must differ from text-only output because
        # position 3 had its embedding replaced by vision features.
        fused_np = ops.convert_to_numpy(output_fused)
        text_np = ops.convert_to_numpy(output_text)
        self.assertFalse(
            np.allclose(fused_np, text_np, atol=1e-5),
            "Vision fusion should change the output.",
        )

    def test_audio_encoder_integration(self):
        """Attach a tiny audio encoder and verify it is stored."""
        from keras_hub.src.models.qwen3_omni.qwen3_omni_audio_encoder import (
            Qwen3OmniAudioEncoder,
        )

        audio_encoder = Qwen3OmniAudioEncoder(
            num_mel_bins=80,
            d_model=32,
            encoder_layers=2,
            encoder_attention_heads=4,
            encoder_ffn_dim=64,
            output_dim=16,
            max_source_positions=100,
            scale_embedding=False,
            dtype="float32",
        )
        init_kwargs = self.init_kwargs.copy()
        init_kwargs["audio_encoder"] = audio_encoder
        model = Qwen3OmniBackbone(**init_kwargs)

        self.assertIsNotNone(model.audio_encoder)
        self.assertIsNone(model.vision_encoder)
        self.assertEqual(model.audio_encoder.output_dim, 16)
