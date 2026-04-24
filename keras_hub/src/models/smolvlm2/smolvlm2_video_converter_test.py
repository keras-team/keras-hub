import numpy as np
from keras import ops

from keras_hub.src.models.smolvlm2.smolvlm2_video_converter import (
    SmolVLM2VideoConverter,
)


def _make_converter(**overrides):
    """Create a small SmolVLM2VideoConverter for testing."""
    kwargs = {
        "max_image_size": 32,
        "size": 64,
        "num_frames": 4,
        "fps": 1,
        "scale": [1 / 255.0] * 3,
        "offset": [0.0] * 3,
        "interpolation": "bicubic",
    }
    kwargs.update(overrides)
    return SmolVLM2VideoConverter(**kwargs)


def _make_video(num_frames=8, height=48, width=64):
    """Create a random video tensor (T, H, W, 3)."""
    return np.random.randint(
        0, 256, size=(num_frames, height, width, 3)
    ).astype("uint8")


class TestSmolVLM2VideoConverter:
    def test_output_shape(self):
        """Output should be (num_sampled_frames, ms, ms, 3)."""
        converter = _make_converter()
        video = _make_video(num_frames=8)
        result = converter(video)
        pixel_values = result["pixel_values"]
        # num_frames=4, so 4 frames sampled from 8
        assert pixel_values.shape == (4, 32, 32, 3), (
            f"Got shape {pixel_values.shape}"
        )

    def test_fewer_frames_than_max(self):
        """When video has fewer frames than num_frames, use all."""
        converter = _make_converter(num_frames=10)
        video = _make_video(num_frames=3)
        result = converter(video)
        num_frames = int(result["num_frames"])
        assert num_frames == 3, f"Expected 3 frames, got {num_frames}"
        assert result["pixel_values"].shape[0] == 3

    def test_single_frame_video(self):
        """Single-frame video should produce one output frame."""
        converter = _make_converter()
        video = _make_video(num_frames=1)
        result = converter(video)
        assert int(result["num_frames"]) == 1
        assert result["pixel_values"].shape == (1, 32, 32, 3)

    def test_normalization_range(self):
        """Output should be normalized (not raw 0-255)."""
        converter = _make_converter()
        video = _make_video(num_frames=4)
        result = converter(video)
        pixel_values = ops.convert_to_numpy(result["pixel_values"])
        # With scale=1/255 and offset=0, max should be ≈ 1.0
        assert pixel_values.max() <= 1.01, (
            f"Max pixel value {pixel_values.max()} exceeds 1.0"
        )

    def test_output_no_nan(self):
        """Output should not contain NaN or Inf."""
        converter = _make_converter()
        video = _make_video(num_frames=4)
        result = converter(video)
        pixel_values = ops.convert_to_numpy(result["pixel_values"])
        assert not np.any(np.isnan(pixel_values)), "NaN in output"
        assert not np.any(np.isinf(pixel_values)), "Inf in output"

    def test_num_frames_output(self):
        """num_frames in output should match sampled count."""
        converter = _make_converter(num_frames=4)
        video = _make_video(num_frames=8)
        result = converter(video)
        assert int(result["num_frames"]) == 4

    def test_config_roundtrip(self):
        """get_config should return all constructor arguments."""
        converter = _make_converter(
            max_image_size=64,
            size=128,
            num_frames=16,
            fps=2,
        )
        cfg = converter.get_config()
        assert cfg["max_image_size"] == 64
        assert cfg["size"] == 128
        assert cfg["num_frames"] == 16
        assert cfg["fps"] == 2
