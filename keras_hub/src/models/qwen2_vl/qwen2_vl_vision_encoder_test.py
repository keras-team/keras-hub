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

        # Derive patch dimensions from init_kwargs to avoid drift.
        kw = self.init_kwargs
        patch_flat_dim = (
            kw["in_channels"]
            * kw["temporal_patch_size"]
            * kw["patch_size"] ** 2
        )

        # 1 image with t=2, h=2, w=2 → total_patches = 8
        grid_thw = np.array([[2, 2, 2]], dtype="int32")
        total_patches = int(np.prod(grid_thw))
        hidden_states = np.random.rand(total_patches, patch_flat_dim).astype(
            "float32"
        )

        output = encoder(hidden_states, grid_thw)

        # After merger, should reduce by spatial_merge_size^2
        merge_sq = kw["spatial_merge_size"] ** 2
        expected_tokens = total_patches // merge_sq
        self.assertEqual(output.shape, (expected_tokens, kw["hidden_size"]))

    def test_vision_encoder_config_roundtrip(self):
        encoder = Qwen2VLVisionEncoder(**self.init_kwargs)
        config = encoder.get_config()
        new_encoder = Qwen2VLVisionEncoder.from_config(config)

        # Verify config values match
        self.assertEqual(encoder.patch_size, new_encoder.patch_size)
        self.assertEqual(
            encoder.temporal_patch_size, new_encoder.temporal_patch_size
        )
        self.assertEqual(encoder.in_channels, new_encoder.in_channels)
        self.assertEqual(encoder.embed_dim, new_encoder.embed_dim)
        self.assertEqual(encoder.hidden_size, new_encoder.hidden_size)
        self.assertEqual(encoder.depth, new_encoder.depth)
        self.assertEqual(encoder.num_heads, new_encoder.num_heads)
        self.assertEqual(encoder.mlp_ratio, new_encoder.mlp_ratio)
        self.assertEqual(
            encoder.spatial_merge_size, new_encoder.spatial_merge_size
        )

    @pytest.mark.large
    def test_vision_encoder_with_multiple_images(self):
        encoder = Qwen2VLVisionEncoder(**self.init_kwargs)

        kw = self.init_kwargs
        patch_flat_dim = (
            kw["in_channels"]
            * kw["temporal_patch_size"]
            * kw["patch_size"] ** 2
        )

        # 2 images with different grid sizes
        grid_thw = np.array([[2, 2, 2], [2, 4, 4]], dtype="int32")
        total_patches = int(np.sum(np.prod(grid_thw, axis=1)))
        hidden_states = np.random.rand(total_patches, patch_flat_dim).astype(
            "float32"
        )

        output = encoder(hidden_states, grid_thw)

        merge_sq = kw["spatial_merge_size"] ** 2
        expected_tokens = total_patches // merge_sq
        self.assertEqual(output.shape, (expected_tokens, kw["hidden_size"]))

    def test_rotary_embeddings(self):
        encoder = Qwen2VLVisionEncoder(**self.init_kwargs)

        # Test that rotary embeddings are generated correctly
        grid_thw = np.array([[1, 2, 2]], dtype="int32")
        cos, sin = encoder._rot_pos_emb(grid_thw)

        # Should have embeddings for all tokens
        # 1 * 2 * 2 = 4 patches total
        self.assertEqual(cos.shape[0], 4)
        self.assertEqual(sin.shape[0], 4)
