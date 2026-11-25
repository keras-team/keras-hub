import numpy as np
from keras import ops

from keras_hub.src.models.qwen2_vl.qwen2_vl_image_converter import (
    Qwen2VLImageConverter,
)
from keras_hub.src.models.qwen2_vl.qwen2_vl_video_converter import (
    Qwen2VLVideoConverter,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen2VLVideoConverterTest(TestCase):
    # Subclass tests (Qwen2.5-VL) swap these implementations.
    video_converter_cls = Qwen2VLVideoConverter
    image_converter_cls = Qwen2VLImageConverter

    def _make_converter(self, **overrides):
        """Create a small video converter for testing."""
        kwargs = {
            "min_pixels": 56 * 56,
            "max_pixels": 56 * 56 * 4,
        }
        kwargs.update(overrides)
        return self.video_converter_cls(
            image_converter=self.image_converter_cls(
                patch_size=14,
                temporal_patch_size=2,
                merge_size=2,
            ),
            **kwargs,
        )

    def _make_video(self, num_frames=4, height=56, width=56):
        """Create a random video tensor (T, H, W, 3)."""
        rng = np.random.default_rng(0)
        return rng.integers(0, 256, size=(num_frames, height, width, 3)).astype(
            "float32"
        )

    def test_output_shape(self):
        """Patches are (total_patches, tps, p, p, C); grid_thw is (1, 3)."""
        converter = self._make_converter()
        result = converter(self._make_video(num_frames=4))
        # 56x56 frames → 4x4 patch grid, tps=2 → grid_t=2, 2*4*4=32 patches.
        self.assertEqual(result["patches"].shape, (32, 2, 14, 14, 3))
        self.assertAllEqual(result["grid_thw"], [[2, 4, 4]])

    def test_odd_num_frames_pads_to_temporal_patch(self):
        """An odd frame count is padded up to `temporal_patch_size`."""
        converter = self._make_converter()
        result = converter(self._make_video(num_frames=3))
        self.assertEqual(result["patches"].shape, (32, 2, 14, 14, 3))
        self.assertAllEqual(result["grid_thw"], [[2, 4, 4]])

    def test_output_no_nan(self):
        converter = self._make_converter()
        result = converter(self._make_video())
        patches = ops.convert_to_numpy(result["patches"])
        self.assertTrue(np.all(np.isfinite(patches)))

    def test_normalization(self):
        """Normalized patches should match the image converter's output."""
        converter = self._make_converter()
        video = self._make_video()
        result = converter(video)
        expected = self.image_converter_cls(
            patch_size=14,
            temporal_patch_size=2,
            merge_size=2,
        )(video, min_pixels=56 * 56, max_pixels=56 * 56 * 4)
        self.assertAllClose(result["patches"], expected["patches"])
        self.assertAllEqual(result["grid_thw"], expected["grid_thw"])

    def test_list_of_videos(self):
        """A list of videos returns a single merged dict."""
        converter = self._make_converter()
        result = converter(
            [self._make_video(num_frames=4), self._make_video(num_frames=2)]
        )
        self.assertEqual(result["patches"].shape[0], 32 + 16)
        self.assertAllEqual(result["grid_thw"], [[2, 4, 4], [1, 4, 4]])

    def test_tuple_of_videos(self):
        """A tuple of clips is processed like a list."""
        converter = self._make_converter()
        result = converter(
            (self._make_video(num_frames=4), self._make_video(num_frames=2))
        )
        self.assertEqual(result["patches"].shape[0], 32 + 16)
        self.assertAllEqual(result["grid_thw"], [[2, 4, 4], [1, 4, 4]])

    def test_5d_batched_tensor(self):
        """A (B, T, H, W, C) batch splits into B separate clips."""
        converter = self._make_converter()
        batched = np.stack(
            [self._make_video(num_frames=4), self._make_video(num_frames=4)],
            axis=0,
        )
        result = converter(batched)
        self.assertEqual(result["patches"].shape[0], 32 + 32)
        self.assertAllEqual(result["grid_thw"], [[2, 4, 4], [2, 4, 4]])

    def test_nested_list_is_one_clip(self):
        """A nested Python list of frames is one clip, not a collection."""
        converter = self._make_converter()
        clip = self._make_video(num_frames=4)
        result = converter(clip.tolist())
        self.assertAllEqual(result["grid_thw"], [[2, 4, 4]])

    def test_nested_list_with_cap(self):
        """Nested lists also work through the `cap_pixels_per_frame`
        branch (was an AttributeError before `.shape` conversion)."""
        converter = self._make_converter(
            cap_pixels_per_frame=True, max_video_tokens=32
        )
        clip = self._make_video(num_frames=4, height=200, width=200)
        result = converter(clip.tolist())
        self.assertAllEqual(result["grid_thw"], [[2, 6, 6]])

    def test_3d_clip_is_one_frame(self):
        """A bare (H, W, C) frame is a single-frame clip."""
        converter = self._make_converter()
        frame = self._make_video(num_frames=4)[0]
        result = converter(frame)
        self.assertAllEqual(result["grid_thw"], [[1, 4, 4]])

    def test_empty_collection_raises(self):
        converter = self._make_converter()
        with self.assertRaises(ValueError):
            converter([])
        with self.assertRaises(ValueError):
            converter(())

    def test_default_video_bounds(self):
        """A default converter applies image bounds, matching HF
        ``from_pretrained`` which maps image `min_pixels`/`max_pixels`
        onto the video processor's size."""
        converter = self.video_converter_cls()
        self.assertEqual(converter.min_pixels, 3136)
        self.assertEqual(converter.max_pixels, 12845056)
        # 56x56 frames sit exactly on the 3136 floor and are not resized.
        result = converter(self._make_video(num_frames=4))
        self.assertAllEqual(result["grid_thw"], [[2, 4, 4]])

    def test_cap_pixels_per_frame(self):
        """The cap shrinks frames once the token budget is exceeded."""
        video = self._make_video(num_frames=4, height=200, width=200)
        uncapped = self._make_converter()(video)
        capped = self._make_converter(
            cap_pixels_per_frame=True, max_video_tokens=32
        )(video)
        self.assertAllEqual(uncapped["grid_thw"], [[2, 8, 8]])
        # 32*28*28*0.9*2 // 4 frames = 11289 px/frame budget → 84x84 px.
        self.assertAllEqual(capped["grid_thw"], [[2, 6, 6]])

    def test_config_roundtrip(self):
        converter = self._make_converter(max_pixels=56 * 56 * 8)
        self.run_serialization_test(converter)
        cfg = converter.get_config()
        self.assertEqual(cfg["max_pixels"], 56 * 56 * 8)
        self.assertEqual(cfg["min_pixels"], 56 * 56)
        self.assertFalse(cfg["cap_pixels_per_frame"])
        image_cfg = cfg["image_converter"]["config"]
        self.assertEqual(image_cfg["patch_size"], 14)
        self.assertEqual(image_cfg["temporal_patch_size"], 2)
