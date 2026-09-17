import numpy as np
from keras import ops

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
            "out_dim": 128,
            "depth": 2,
            "num_heads": 4,
            "mlp_ratio": 4,
            "spatial_merge_size": 2,
        }

    def _patches(self, n_patches, seed=0):
        rng = np.random.default_rng(seed)
        kw = self.init_kwargs
        return rng.random(
            (
                n_patches,
                kw["temporal_patch_size"],
                kw["patch_size"],
                kw["patch_size"],
                kw["in_channels"],
            )
        ).astype("float32")

    def test_vision_encoder_basics(self):
        encoder = Qwen2VLVisionEncoder(**self.init_kwargs)

        # 1 image with t=2, h=2, w=2 → total_patches = 8
        grid_thw = np.array([[2, 2, 2]], dtype="int32")
        total_patches = int(np.prod(grid_thw))
        hidden_states = self._patches(total_patches)

        output = encoder(hidden_states, grid_thw)

        # After merger, should reduce by spatial_merge_size^2
        merge_sq = self.init_kwargs["spatial_merge_size"] ** 2
        expected_tokens = total_patches // merge_sq
        self.assertEqual(
            tuple(output.shape), (expected_tokens, self.init_kwargs["out_dim"])
        )

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
        self.assertEqual(encoder.out_dim, new_encoder.out_dim)
        self.assertEqual(encoder.depth, new_encoder.depth)
        self.assertEqual(encoder.num_heads, new_encoder.num_heads)
        self.assertEqual(encoder.mlp_ratio, new_encoder.mlp_ratio)
        self.assertEqual(
            encoder.spatial_merge_size, new_encoder.spatial_merge_size
        )

    def test_vision_encoder_with_multiple_images(self):
        encoder = Qwen2VLVisionEncoder(**self.init_kwargs)

        # 2 images with different grid sizes
        grid_thw = np.array([[2, 2, 2], [2, 4, 4]], dtype="int32")
        total_patches = int(np.sum(np.prod(grid_thw, axis=1)))
        hidden_states = self._patches(total_patches)

        output = encoder(hidden_states, grid_thw)

        merge_sq = self.init_kwargs["spatial_merge_size"] ** 2
        expected_tokens = total_patches // merge_sq
        self.assertEqual(
            tuple(output.shape), (expected_tokens, self.init_kwargs["out_dim"])
        )

    def test_per_frame_attention_isolation(self):
        """Attention is confined per (image, frame): concatenating two
        images must produce the same per-image outputs as processing them
        separately."""
        encoder = Qwen2VLVisionEncoder(**self.init_kwargs)

        grid_a = np.array([[1, 2, 2]], dtype="int32")
        grid_b = np.array([[1, 4, 4]], dtype="int32")
        patches_a = self._patches(int(np.prod(grid_a)), seed=1)
        patches_b = self._patches(int(np.prod(grid_b)), seed=2)

        out_a = ops.convert_to_numpy(encoder(patches_a, grid_a))
        out_b = ops.convert_to_numpy(encoder(patches_b, grid_b))

        joint = encoder(
            np.concatenate([patches_a, patches_b], axis=0),
            np.concatenate([grid_a, grid_b], axis=0),
        )
        joint = ops.convert_to_numpy(joint)

        n_a = out_a.shape[0]
        np.testing.assert_allclose(joint[:n_a], out_a, atol=1e-5)
        np.testing.assert_allclose(joint[n_a:], out_b, atol=1e-5)

    def test_empty_input(self):
        """Zero-patch input returns an empty output (text-only path)."""
        encoder = Qwen2VLVisionEncoder(**self.init_kwargs)
        kw = self.init_kwargs
        patches = np.zeros(
            (
                0,
                kw["temporal_patch_size"],
                kw["patch_size"],
                kw["patch_size"],
                kw["in_channels"],
            ),
            dtype="float32",
        )
        grid_thw = np.zeros((0, 3), dtype="int32")
        output = encoder(patches, grid_thw)
        self.assertEqual(tuple(output.shape), (0, kw["out_dim"]))

    def test_rotary_embeddings(self):
        encoder = Qwen2VLVisionEncoder(**self.init_kwargs)

        # grid [1, 2, 2] → 4 patches, head_dim = 64/4 = 16.
        grid_thw = np.array([[1, 2, 2]], dtype="int32")
        cos, sin = encoder._rot_pos_emb(grid_thw)

        self.assertEqual(tuple(cos.shape), (4, 16))
        self.assertEqual(tuple(sin.shape), (4, 16))
