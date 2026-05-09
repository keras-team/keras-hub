import numpy as np
from keras import ops

from keras_hub.src.models.smolvlm2.smolvlm2_vision_encoder import (
    SmolVLM2VisionEncoder,
)


def _make_encoder(**overrides):
    """Create a small SmolVLM2VisionEncoder for testing."""
    kwargs = {
        "image_size": 32,
        "patch_size": 16,
        "hidden_dim": 64,
        "intermediate_dim": 128,
        "num_layers": 2,
        "num_heads": 4,
        "num_channels": 3,
        "layer_norm_epsilon": 1e-6,
    }
    kwargs.update(overrides)
    return SmolVLM2VisionEncoder(**kwargs)


class TestSmolVLM2VisionEncoder:
    def test_output_shape(self):
        """Single image produces (1, num_patches, hidden_dim) output."""
        encoder = _make_encoder()
        # image_size=32, patch_size=16 → 2×2 = 4 patches.
        pixel_values = np.random.rand(1, 32, 32, 3).astype("float32")
        output = encoder({"pixel_values": pixel_values})
        assert output.shape == (1, 4, 64), f"Got shape {output.shape}"

    def test_batch_output_shape(self):
        """Batch of images produces correct shape."""
        encoder = _make_encoder()
        pixel_values = np.random.rand(3, 32, 32, 3).astype("float32")
        output = encoder({"pixel_values": pixel_values})
        assert output.shape == (3, 4, 64), f"Got shape {output.shape}"

    def test_output_does_not_contain_nan(self):
        """Forward pass should produce finite outputs."""
        encoder = _make_encoder()
        pixel_values = np.random.rand(1, 32, 32, 3).astype("float32")
        output = encoder({"pixel_values": pixel_values})
        output_np = ops.convert_to_numpy(output)
        assert not np.any(np.isnan(output_np)), "NaN in vision encoder output"
        assert not np.any(np.isinf(output_np)), "Inf in vision encoder output"

    def test_num_patches_formula(self):
        """Verify num_patches = (image_size / patch_size)²."""
        for image_size, patch_size, expected_patches in [
            (32, 16, 4),
            (32, 8, 16),
            (64, 16, 16),
        ]:
            encoder = _make_encoder(
                image_size=image_size, patch_size=patch_size
            )
            pixel_values = np.random.rand(1, image_size, image_size, 3).astype(
                "float32"
            )
            output = encoder({"pixel_values": pixel_values})
            assert output.shape[1] == expected_patches, (
                f"image_size={image_size}, patch_size={patch_size}: "
                f"expected {expected_patches} patches, got {output.shape[1]}"
            )

    def test_get_config_roundtrip(self):
        """get_config should return all constructor arguments."""
        encoder = _make_encoder(
            image_size=64,
            patch_size=8,
            hidden_dim=128,
            intermediate_dim=256,
            num_layers=3,
            num_heads=8,
        )
        cfg = encoder.get_config()
        assert cfg["image_size"] == 64
        assert cfg["patch_size"] == 8
        assert cfg["hidden_dim"] == 128
        assert cfg["intermediate_dim"] == 256
        assert cfg["num_layers"] == 3
        assert cfg["num_heads"] == 8
        assert cfg["num_channels"] == 3
        assert cfg["layer_norm_epsilon"] == 1e-6

    def test_different_hidden_dim(self):
        """Encoder with different hidden_dim produces correct output dim."""
        encoder = _make_encoder(hidden_dim=128, intermediate_dim=256)
        pixel_values = np.random.rand(1, 32, 32, 3).astype("float32")
        output = encoder({"pixel_values": pixel_values})
        assert output.shape == (1, 4, 128), f"Got shape {output.shape}"

    def test_parameter_count_positive(self):
        """Encoder should have a non-trivial number of parameters."""
        encoder = _make_encoder()
        assert encoder.count_params() > 0
