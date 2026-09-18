import numpy as np
from keras import ops

from keras_hub.src.models.smolvlm2.smolvlm2_video_converter import (
    SmolVLM2VideoConverter,
)
from keras_hub.src.tests.test_case import TestCase


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


class SmolVLM2VideoConverterTest(TestCase):
    def test_output_shape(self):
        """Output should be (num_sampled_frames, ms, ms, 3)."""
        converter = _make_converter()
        result = converter(_make_video(num_frames=8))
        # num_frames=4, so 4 frames sampled from 8.
        self.assertEqual(result["pixel_values"].shape, (4, 32, 32, 3))

    def test_fewer_frames_than_max(self):
        """When video has fewer frames than num_frames, use all."""
        converter = _make_converter(num_frames=10)
        result = converter(_make_video(num_frames=3))
        self.assertEqual(int(result["num_frames"]), 3)
        self.assertEqual(result["pixel_values"].shape[0], 3)

    def test_single_frame_video(self):
        """Single-frame video should produce one output frame."""
        converter = _make_converter()
        result = converter(_make_video(num_frames=1))
        self.assertEqual(int(result["num_frames"]), 1)
        self.assertEqual(result["pixel_values"].shape, (1, 32, 32, 3))

    def test_normalization_range(self):
        """Output should be normalized (not raw 0-255)."""
        converter = _make_converter()
        result = converter(_make_video(num_frames=4))
        pixel_values = ops.convert_to_numpy(result["pixel_values"])
        # With scale=1/255 and offset=0, max should be ≈ 1.0.
        self.assertAllInRange(pixel_values, -0.01, 1.01)

    def test_output_no_nan(self):
        """Output should not contain NaN or Inf."""
        converter = _make_converter()
        result = converter(_make_video(num_frames=4))
        pixel_values = ops.convert_to_numpy(result["pixel_values"])
        self.assertTrue(np.all(np.isfinite(pixel_values)))

    def test_num_frames_output(self):
        """num_frames in output should match sampled count."""
        converter = _make_converter(num_frames=4)
        result = converter(_make_video(num_frames=8))
        self.assertEqual(int(result["num_frames"]), 4)

    def test_batched_video_raises(self):
        """A 5-D batch of videos is not supported and must raise."""
        converter = _make_converter()
        videos = np.stack([_make_video(num_frames=4)] * 2, axis=0)
        with self.assertRaisesRegex(ValueError, "single video"):
            converter(videos)

    def test_config_roundtrip(self):
        """get_config should return all constructor arguments."""
        converter = _make_converter(
            max_image_size=64,
            size=128,
            num_frames=16,
            fps=2,
            interpolation="bilinear",
            antialias=False,
        )
        self.run_serialization_test(converter)
        cfg = converter.get_config()
        self.assertEqual(cfg["max_image_size"], 64)
        self.assertEqual(cfg["size"], 128)
        self.assertEqual(cfg["num_frames"], 16)
        self.assertEqual(cfg["fps"], 2)
        # Previously dropped from the config, silently resetting to the
        # `bicubic`/`True` defaults on reload.
        self.assertEqual(cfg["interpolation"], "bilinear")
        self.assertFalse(cfg["antialias"])

    def test_frame_converter_alias(self):
        """`frame_converter` mirrors the converter used by `call()`."""
        converter = _make_converter()
        self.assertIs(converter.frame_converter, converter.image_converter)
