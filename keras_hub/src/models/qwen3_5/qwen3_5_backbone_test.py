import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.qwen3_5.qwen3_5_backbone import Qwen3_5Backbone
from keras_hub.src.models.qwen3_5.qwen3_5_vision_encoder import (
    Qwen3_5VisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase


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
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=Qwen3_5Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 16),
            run_quantization_check=False,
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
        # Just verify the model builds and has params.
        self.assertGreater(model.count_params(), 0)


class Qwen3_5MultimodalBackboneTest(TestCase):
    """Tests for the backbone with vision encoder attached."""

    def setUp(self):
        # Use a tiny vision encoder for testing.
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
        # Should have vision encoder stored as attribute.
        self.assertIsNotNone(model.vision_encoder)
        self.assertTrue(hasattr(model, "interleave_embeddings"))

    def test_multimodal_backbone_text_only_forward(self):
        """Multimodal backbone should still work with text-only inputs."""
        model = Qwen3_5Backbone(**self.init_kwargs)
        input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }
        output = model(input_data)
        self.assertEqual(ops.shape(output), (2, 5, 16))

    def test_vision_encoder_standalone(self):
        """Test vision encoder produces correct output shape standalone."""
        encoder = self.vision_encoder

        # For a 16x16 image with patch_size=4 and temporal_patch_size=2:
        # Grid in patch units: T=1, H=4, W=4
        # Total patches = 1*4*4 = 16
        # After spatial merge (size=2): (4/2)*(4/2) = 4 tokens per frame
        # Total merged tokens = 1 * 4 = 4
        grid_thw = np.array([[1, 4, 4]], dtype="int32")
        total_patches = 1 * 4 * 4
        pixel_values = np.random.randn(total_patches, 2, 4, 4, 3).astype(
            "float32"
        )

        output = encoder(
            ops.convert_to_tensor(pixel_values),
            ops.convert_to_tensor(grid_thw),
        )
        # Should produce (4 merged tokens, hidden_size=16)
        self.assertEqual(ops.shape(output), (4, 16))

    def test_interleave_embeddings(self):
        """Test that interleave layer correctly scatters vision tokens."""
        model = Qwen3_5Backbone(**self.init_kwargs)

        batch_size = 1
        seq_len = 8
        hidden_dim = 16

        # Create text embeddings (all zeros).
        text_emb = ops.zeros((batch_size, seq_len, hidden_dim))

        # Create vision embeddings (all ones for 2 tokens).
        vision_emb = ops.ones((2, hidden_dim))

        # Place at positions 1 and 3 in the sequence.
        indices = ops.convert_to_tensor([1, 3], dtype="int32")

        result = model.interleave_embeddings(
            image_embeddings=vision_emb,
            text_embeddings=text_emb,
            vision_indices=indices,
        )
        self.assertEqual(ops.shape(result), (batch_size, seq_len, hidden_dim))

        # Positions 1 and 3 should have been replaced with ones.
        try:
            result_np = np.array(result)
        except TypeError:
            result_np = np.array(result.cpu().detach())

        # Position 0 should still be zero.
        np.testing.assert_allclose(result_np[0, 0, :], 0.0, atol=1e-6)
        # Position 1 should be ones.
        np.testing.assert_allclose(result_np[0, 1, :], 1.0, atol=1e-6)
        # Position 2 should still be zero.
        np.testing.assert_allclose(result_np[0, 2, :], 0.0, atol=1e-6)
        # Position 3 should be ones.
        np.testing.assert_allclose(result_np[0, 3, :], 1.0, atol=1e-6)
