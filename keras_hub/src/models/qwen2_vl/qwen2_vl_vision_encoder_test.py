import numpy as np
import pytest

from keras_hub.src.models.qwen2_vl.qwen2_vl_vision_encoder import (
    Qwen2VLVisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen2VLVisionEncoderTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "patch_size": 14,
            "temporal_patch_size": 2,
            "in_channels": 3,
            "embed_dim": 64,
            "hidden_size": 128,
            "depth": 2,
            "num_heads": 4,
            "mlp_ratio": 4,
            "spatial_merge_size": 2,
        }

    def test_vision_encoder_basics(self):
        encoder = Qwen2VLVisionEncoder(**self.init_kwargs)

        # Create dummy flat patch input
        # (total_patches, in_channels * temporal_patch_size * patch_size^2)
        # total_patches must equal grid_t * grid_h * grid_w
        patch_flat_dim = 3 * 2 * 14 * 14
        # grid_thw = [2, 2, 2] → total_patches = 2 * 2 * 2 = 8
        hidden_states = np.random.rand(8, patch_flat_dim).astype("float32")

        # Create dummy grid_thw (1 image with t=2, h=2, w=2)
        grid_thw = np.array([[2, 2, 2]], dtype="int32")

        output = encoder(hidden_states, grid_thw)

        # After merger, should reduce by spatial_merge_size^2
        expected_tokens = 8 // (2**2)  # 8 / 4 = 2
        self.assertEqual(output.shape, (expected_tokens, 128))

    def test_vision_encoder_config_roundtrip(self):
        encoder = Qwen2VLVisionEncoder(**self.init_kwargs)
        config = encoder.get_config()
        new_encoder = Qwen2VLVisionEncoder.from_config(config)

        # Verify config values match
        self.assertEqual(encoder.patch_size, new_encoder.patch_size)
        self.assertEqual(encoder.embed_dim, new_encoder.embed_dim)
        self.assertEqual(encoder.hidden_size, new_encoder.hidden_size)
        self.assertEqual(encoder.depth, new_encoder.depth)

    @pytest.mark.large
    def test_vision_encoder_with_multiple_images(self):
        encoder = Qwen2VLVisionEncoder(**self.init_kwargs)

        # 2 images with different grid sizes
        patch_flat_dim = 3 * 2 * 14 * 14
        # Image 1: 2x2x2 = 8 patches, Image 2: 2x4x4 = 32 patches
        total_patches = 8 + 32
        hidden_states = np.random.rand(total_patches, patch_flat_dim).astype(
            "float32"
        )
        grid_thw = np.array([[2, 2, 2], [2, 4, 4]], dtype="int32")

        output = encoder(hidden_states, grid_thw)

        # After merger: 8/4 + 32/4 = 2 + 8 = 10 tokens
        expected_tokens = (8 + 32) // (2**2)
        self.assertEqual(output.shape, (expected_tokens, 128))

    def test_rotary_embeddings(self):
        encoder = Qwen2VLVisionEncoder(**self.init_kwargs)

        # Test that rotary embeddings are generated correctly
        grid_thw = np.array([[1, 2, 2]], dtype="int32")
        cos, sin = encoder._rot_pos_emb(grid_thw)

        # Should have embeddings for all tokens
        # 1 * 2 * 2 = 4 patches total
        self.assertEqual(cos.shape[0], 4)
        self.assertEqual(sin.shape[0], 4)
