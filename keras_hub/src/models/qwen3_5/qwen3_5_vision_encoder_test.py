"""Tests for Qwen3_5VisionEncoder."""

import numpy as np

from keras_hub.src.models.qwen3_5.qwen3_5_vision_encoder import (
    Qwen3_5VisionEncoder,
)


def _make_encoder(out_hidden_size=64):
    return Qwen3_5VisionEncoder(
        depth=2,
        hidden_size=32,
        num_heads=4,
        intermediate_size=64,
        in_channels=3,
        patch_size=4,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=out_hidden_size,
        num_position_embeddings=16,  # 4×4 grid
    )


def _make_pixel_values(t, h, w, patch_size=4, temporal_patch_size=2):
    """Build random patch tensor for (T=t, H=h patches, W=w patches)."""
    num_patches = t * h * w
    return np.random.rand(
        num_patches, temporal_patch_size, patch_size, patch_size, 3
    ).astype("float32")


class TestQwen3_5VisionEncoder:
    def test_output_shape_single_image(self):
        """16 patches, merge=2 → 4 output tokens."""
        encoder = _make_encoder()
        grid_thw = np.array([[1, 4, 4]], dtype="int32")
        pixel_values = _make_pixel_values(1, 4, 4)
        output = encoder(pixel_values, grid_thw)
        # 1*4*4 // 2^2 = 4 merged tokens
        assert output.shape == (4, 64), f"Got shape {output.shape}"

    def test_output_shape_two_images(self):
        """Two images concatenated → 4+4=8 tokens."""
        encoder = _make_encoder()
        grid_thw = np.array([[1, 4, 4], [1, 4, 4]], dtype="int32")
        # Stack patches for both images.
        pv1 = _make_pixel_values(1, 4, 4)
        pv2 = _make_pixel_values(1, 4, 4)
        pixel_values = np.concatenate([pv1, pv2], axis=0)
        output = encoder(pixel_values, grid_thw)
        assert output.shape[0] == 8, f"Expected 8 tokens, got {output.shape[0]}"

    def test_token_count_formula(self):
        """Verify token count = T × H × W // spatial_merge_size²."""
        encoder = _make_encoder()
        for t, h, w in [(1, 4, 4), (1, 4, 8), (1, 8, 4)]:
            grid_thw = np.array([[t, h, w]], dtype="int32")
            pixel_values = _make_pixel_values(t, h, w)
            output = encoder(pixel_values, grid_thw)
            expected = (t * h * w) // 4  # spatial_merge_size=2
            assert output.shape[0] == expected, (
                f"Grid ({t},{h},{w}): "
                f"expected {expected}, got {output.shape[0]}"
            )

    def test_output_does_not_contain_nan(self):
        """Forward pass should produce finite outputs."""
        encoder = _make_encoder()
        grid_thw = np.array([[1, 4, 4]], dtype="int32")
        pixel_values = _make_pixel_values(1, 4, 4)
        output = encoder(pixel_values, grid_thw)
        # Handle PyTorch MPS tensors that can't be converted to numpy directly.
        try:
            output_np = np.array(output)
        except TypeError:
            # PyTorch MPS device: move to CPU first.
            output_np = np.array(output.cpu().detach())
        assert not np.any(np.isnan(output_np)), "NaN in vision encoder output"
        assert not np.any(np.isinf(output_np)), "Inf in vision encoder output"

    def test_get_config_roundtrip(self):
        """get_config should return all expected constructor arguments."""
        encoder = _make_encoder(out_hidden_size=128)
        cfg = encoder.get_config()
        assert cfg["depth"] == 2
        assert cfg["hidden_size"] == 32
        assert cfg["num_heads"] == 4
        assert cfg["patch_size"] == 4
        assert cfg["spatial_merge_size"] == 2
        assert cfg["out_hidden_size"] == 128
        assert cfg["num_position_embeddings"] == 16
