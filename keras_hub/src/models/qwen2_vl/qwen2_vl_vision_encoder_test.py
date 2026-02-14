"""Tests for Qwen2-VL Vision Encoder components."""

import numpy as np
from keras import ops

from keras_hub.src.models.qwen2_vl.qwen2_vl_vision_encoder import (
    Qwen2VLPatchEmbed,
    Qwen2VLPatchMerger,
    Qwen2VLVisionBlock,
    Qwen2VLVisionEncoder,
    Qwen2VLVisionRotaryEmbedding,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen2VLPatchEmbedTest(TestCase):
    def test_output_shape(self):
        patch_embed = Qwen2VLPatchEmbed(
            patch_size=4,
            temporal_patch_size=2,
            in_channels=3,
            embed_dim=32,
        )
        # Input: (batch, in_channels, temporal, patch_h, patch_w)
        dummy_input = np.random.rand(8, 3, 2, 4, 4).astype("float32")
        output = patch_embed(dummy_input)
        self.assertEqual(output.shape, (8, 32))


class Qwen2VLVisionRotaryEmbeddingTest(TestCase):
    def test_output_shape(self):
        rope = Qwen2VLVisionRotaryEmbedding(dim=16)
        freqs = rope(seqlen=10)
        self.assertEqual(freqs.shape, (10, 8))


class Qwen2VLVisionBlockTest(TestCase):
    def test_output_shape(self):
        block = Qwen2VLVisionBlock(
            embed_dim=32,
            num_heads=4,
            mlp_ratio=4.0,
        )
        # Input: (seq_len, embed_dim) — no batch dim in vision encoder
        dummy_input = np.random.rand(16, 32).astype("float32")
        # Create dummy position embeddings (shape: seq_len, head_dim)
        head_dim = 32 // 4  # embed_dim // num_heads = 8
        cos = np.ones((16, head_dim), dtype="float32")
        sin = np.zeros((16, head_dim), dtype="float32")
        output = block(dummy_input, position_embeddings=(cos, sin))
        self.assertEqual(output.shape, (16, 32))


class Qwen2VLPatchMergerTest(TestCase):
    def test_output_shape(self):
        merger = Qwen2VLPatchMerger(
            hidden_size=64,
            context_dim=32,
            spatial_merge_size=2,
        )
        # 16 patches, merge 2x2 -> 4 merged patches
        dummy_input = np.random.rand(16, 32).astype("float32")
        output = merger(dummy_input)
        self.assertEqual(output.shape, (4, 64))


class Qwen2VLVisionEncoderTest(TestCase):
    def test_encoder_output_shape(self):
        encoder = Qwen2VLVisionEncoder(
            hidden_size=64,
            embed_dim=32,
            depth=2,
            num_heads=4,
            patch_size=4,
            temporal_patch_size=2,
            in_channels=3,
            mlp_ratio=4.0,
            spatial_merge_size=2,
        )
        # Create input: 1 image with grid_thw = (1, 4, 4)
        # Total patches = 1 * 4 * 4 = 16; after 3D patch embed these
        # become (num_patches, in_channels, temporal, patch_h, patch_w)
        num_patches = 16
        dummy_input = np.random.rand(
            num_patches, 3, 2, 4, 4
        ).astype("float32")
        grid_thw = np.array([[1, 4, 4]], dtype="int32")

        output = encoder(dummy_input, grid_thw)
        # After PatchMerger with spatial_merge_size=2:
        # 16 patches / (2*2) = 4 merged patches
        self.assertEqual(output.shape, (4, 64))

    def test_multi_image_output_shape(self):
        encoder = Qwen2VLVisionEncoder(
            hidden_size=64,
            embed_dim=32,
            depth=2,
            num_heads=4,
            patch_size=4,
            temporal_patch_size=2,
            in_channels=3,
            mlp_ratio=4.0,
            spatial_merge_size=2,
        )
        # 2 images, each with grid_thw (1, 4, 4) -> 16 patches each
        num_patches = 32  # 16 + 16
        dummy_input = np.random.rand(
            num_patches, 3, 2, 4, 4
        ).astype("float32")
        grid_thw = np.array(
            [[1, 4, 4], [1, 4, 4]], dtype="int32"
        )

        output = encoder(dummy_input, grid_thw)
        # 32 patches / 4 = 8 merged patches
        self.assertEqual(output.shape, (8, 64))
