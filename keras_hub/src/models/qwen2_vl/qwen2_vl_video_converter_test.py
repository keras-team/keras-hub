import numpy as np
from keras import ops

from keras_hub.src.models.qwen2_vl.qwen2_vl_image_converter import (
    Qwen2VLImageConverter,
)
from keras_hub.src.models.qwen2_vl.qwen2_vl_video_converter import (
    Qwen2VLVideoConverter,
)
from keras_hub.src.tests.test_case import TestCase


def _make_converter(**overrides):
    """Create a small Qwen2VLVideoConverter for testing."""
    kwargs = {
        "min_pixels": 56 * 56,
        "max_pixels": 56 * 56 * 4,
        "patch_size": 14,
        "temporal_patch_size": 2,
        "merge_size": 2,
    }
    kwargs.update(overrides)
    return Qwen2VLVideoConverter(**kwargs)


def _make_video(num_frames=4, height=56, width=56):
    """Create a random video tensor (T, H, W, 3)."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(num_frames, height, width, 3)).astype(
        "float32"
    )


class Qwen2VLVideoConverterTest(TestCase):
    def test_output_shape(self):
        """Patches are (total_patches, tps, p, p, C); grid_thw is (1, 3)."""
        converter = _make_converter()
        result = converter(_make_video(num_frames=4))
        # 56x56 frames → 4x4 patch grid, tps=2 → grid_t=2, 2*4*4=32 patches.
        self.assertEqual(result["patches"].shape, (32, 2, 14, 14, 3))
        self.assertAllEqual(result["grid_thw"], [[2, 4, 4]])

    def test_odd_num_frames_pads_to_temporal_patch(self):
        """An odd frame count is padded up to `temporal_patch_size`."""
        converter = _make_converter()
        result = converter(_make_video(num_frames=3))
        self.assertEqual(result["patches"].shape, (32, 2, 14, 14, 3))
        self.assertAllEqual(result["grid_thw"], [[2, 4, 4]])

    def test_output_no_nan(self):
        converter = _make_converter()
        result = converter(_make_video())
        patches = ops.convert_to_numpy(result["patches"])
        self.assertTrue(np.all(np.isfinite(patches)))

    def test_normalization(self):
        """Normalized patches should match the image converter's output."""
        converter = _make_converter()
        video = _make_video()
        result = converter(video)
        expected = Qwen2VLImageConverter(
            min_pixels=56 * 56,
            max_pixels=56 * 56 * 4,
            patch_size=14,
            temporal_patch_size=2,
            merge_size=2,
        )(video)
        self.assertAllClose(result["patches"], expected["patches"])
        self.assertAllEqual(result["grid_thw"], expected["grid_thw"])

    def test_list_of_videos(self):
        """A list of videos returns a single merged dict."""
        converter = _make_converter()
        result = converter(
            [_make_video(num_frames=4), _make_video(num_frames=2)]
        )
        self.assertEqual(result["patches"].shape[0], 32 + 16)
        self.assertAllEqual(result["grid_thw"], [[2, 4, 4], [1, 4, 4]])

    def test_config_roundtrip(self):
        converter = _make_converter(max_pixels=56 * 56 * 8)
        self.run_serialization_test(converter)
        cfg = converter.get_config()
        self.assertEqual(cfg["max_pixels"], 56 * 56 * 8)
        self.assertEqual(cfg["patch_size"], 14)
        self.assertEqual(cfg["temporal_patch_size"], 2)
