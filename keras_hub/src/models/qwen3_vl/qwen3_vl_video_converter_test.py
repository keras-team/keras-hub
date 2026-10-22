import grain
import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.qwen3_vl.qwen3_vl_video_converter import (
    Qwen3_VLVideoConverter,
)
from keras_hub.src.models.qwen3_vl.qwen3_vl_video_converter import smart_resize
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.tests.test_case import assert_grain_safe_types

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import torch
    import torchvision  # noqa: F401
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
        Qwen3VLVideoProcessor,
    )
except ImportError:
    torch = None


def _smooth_video(frames, height, width):
    ts, ys, xs = np.mgrid[:frames, :height, :width]
    channels = [
        40 + 150 * ys / height,
        40 + 100 * xs / width + 5 * ts,
        40 + 75 * (ys / height + xs / width),
    ]
    return np.stack(channels, axis=-1).astype("float32")


class Qwen3_VLVideoConverterTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "patch_size": 4,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "min_pixels": 2048,
            "max_pixels": 16384,
        }
        self.converter = Qwen3_VLVideoConverter(**self.init_kwargs)

    def test_defaults(self):
        converter = Qwen3_VLVideoConverter()
        self.assertEqual(converter.interpolation, "bicubic")
        self.assertAllClose(converter.scale, 2.0 / 255.0)
        self.assertAllClose(converter.offset, -1.0)
        self.assertEqual(converter.min_pixels, 4096)
        self.assertEqual(converter.max_pixels, 25165824)
        self.assertEqual(converter.fps, 2.0)
        self.assertEqual(converter.min_frames, 4)
        self.assertEqual(converter.max_frames, 768)

    def test_smart_resize_counts_frames(self):
        # 6 frames of 48x48: volume 13824 fits; 12 frames do not.
        self.assertEqual(smart_resize(6, 48, 48, 2, 8, 2048, 16384), (48, 48))
        self.assertEqual(smart_resize(12, 48, 48, 2, 8, 2048, 16384), (32, 32))
        with self.assertRaises(ValueError):
            smart_resize(4, 4, 48, 2, 8, 2048, 16384)

    def test_output_shapes_and_grid(self):
        outputs = self.converter(np.ones((4, 32, 48, 3), "float32"))
        self.assertEqual(outputs["patches"].shape, (2 * 8 * 12, 2, 4, 4, 3))
        self.assertAllEqual(outputs["grid_thw"], [2, 8, 12])

    def test_odd_frames_repeat_last_frame(self):
        video = _smooth_video(3, 32, 32)
        outputs = self.converter(video)
        self.assertAllEqual(outputs["grid_thw"], [2, 8, 8])
        padded = self.converter(np.concatenate([video, video[-1:]], axis=0))
        self.assertAllClose(outputs["patches"], padded["patches"])

    def test_merge_block_order(self):
        converter = Qwen3_VLVideoConverter(
            patch_size=1,
            spatial_merge_size=2,
            min_pixels=1,
            max_pixels=1000,
            scale=1.0,
            offset=0.0,
        )
        video = np.arange(2 * 16, dtype="float32").reshape(2, 4, 4, 1)
        video = np.repeat(video, 3, axis=-1)
        patches = ops.convert_to_numpy(converter(video)["patches"])
        # Within a patch, the time axis holds frame 0 then frame 1.
        self.assertAllEqual(
            patches[:, 0, 0, 0, 0],
            [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15],
        )
        self.assertAllEqual(patches[:, 1, 0, 0, 0], patches[:, 0, 0, 0, 0] + 16)

    def test_sample_frame_indices(self):
        # 10s at 30 fps sampled at 2 fps gives 20 frames.
        indices = self.converter.sample_frame_indices(300, video_fps=30.0)
        self.assertEqual(len(indices), 20)
        self.assertEqual((indices[0], indices[-1]), (0, 299))
        # Short clips are padded up to min_frames, capped by the clip length.
        self.assertEqual(len(self.converter.sample_frame_indices(48, 24.0)), 4)
        self.assertEqual(len(self.converter.sample_frame_indices(3, 24.0)), 3)
        long = Qwen3_VLVideoConverter(max_frames=8)
        self.assertEqual(len(long.sample_frame_indices(2400, 24.0)), 8)

    def test_get_config_round_trip(self):
        self.run_serialization_test(self.converter)
        config = self.converter.get_config()
        for key, value in self.init_kwargs.items():
            self.assertEqual(config[key], value)
        self.assertEqual(config["fps"], 2.0)

    def test_grain_map(self):
        video = np.ones((4, 32, 48, 3), "uint8") * 7
        expected = self.converter(video)
        ds = grain.MapDataset.source([video, video]).map(self.converter)
        for element in ds:
            assert_grain_safe_types(element)
            self.assertAllClose(element["patches"], expected["patches"])
            self.assertAllEqual(element["grid_thw"], [2, 8, 12])

    @pytest.mark.skipif(tf is None, reason="Requires tensorflow.")
    def test_tf_dataset_map(self):
        video = np.random.RandomState(0).randint(
            0, 256, (3, 70, 50, 3), dtype="uint8"
        )
        ds = tf.data.Dataset.from_tensor_slices([video])
        outputs = next(iter(ds.map(self.converter)))
        eager = self.converter(video)
        self.assertAllEqual(outputs["grid_thw"], eager["grid_thw"])
        # TF and the active backend round a few pixels differently, so allow
        # a one-level difference on a tiny fraction of values.
        diff = np.abs(
            ops.convert_to_numpy(outputs["patches"])
            - ops.convert_to_numpy(eager["patches"])
        )
        self.assertLessEqual(diff.max(), 1.5 / 127.5)
        self.assertLess((diff > 1e-4).mean(), 1e-3)

    @pytest.mark.skipif(
        torch is None, reason="Requires torch, torchvision and transformers."
    )
    def test_matches_hf_video_processor(self):
        hf = Qwen3VLVideoProcessor(
            size={"shortest_edge": 2048, "longest_edge": 16384},
            patch_size=4,
            temporal_patch_size=2,
            merge_size=2,
            image_mean=[0.5] * 3,
            image_std=[0.5] * 3,
            do_sample_frames=False,
        )
        # Aligned, odd frame count, downscaled, and upscaled inputs.
        for shape in [(4, 32, 48), (3, 32, 32), (12, 48, 48), (2, 20, 36)]:
            video = _smooth_video(*shape)
            expected = hf(
                videos=[torch.from_numpy(video).permute(0, 3, 1, 2)],
                return_tensors="np",
                return_metadata=False,
            )
            outputs = self.converter(video)
            pixels = expected["pixel_values_videos"]
            hf_patches = pixels.reshape(pixels.shape[0], 3, 2, 4, 4)
            hf_patches = hf_patches.transpose(0, 2, 3, 4, 1)
            self.assertAllEqual(
                outputs["grid_thw"], expected["video_grid_thw"][0]
            )
            self.assertAllClose(
                outputs["patches"], hf_patches, atol=1e-4, rtol=0
            )

    @pytest.mark.skipif(
        torch is None, reason="Requires torch, torchvision and transformers."
    )
    def test_uint8_matches_hf_video_processor(self):
        hf = Qwen3VLVideoProcessor(
            size={"shortest_edge": 2048, "longest_edge": 16384},
            patch_size=4,
            temporal_patch_size=2,
            merge_size=2,
            image_mean=[0.5] * 3,
            image_std=[0.5] * 3,
            do_sample_frames=False,
        )
        rng = np.random.default_rng(0)
        for shape in [(4, 130, 170), (3, 60, 90)]:
            video = rng.integers(0, 256, (*shape, 3)).astype("uint8")
            expected = hf(
                videos=[torch.from_numpy(video).permute(0, 3, 1, 2)],
                return_tensors="np",
                return_metadata=False,
            )
            outputs = self.converter(video)
            pixels = expected["pixel_values_videos"]
            hf_patches = pixels.reshape(pixels.shape[0], 3, 2, 4, 4)
            hf_patches = hf_patches.transpose(0, 2, 3, 4, 1)
            diff = np.abs(ops.convert_to_numpy(outputs["patches"]) - hf_patches)
            self.assertLess(diff.mean(), 2e-3)
            self.assertLess((diff > 2.0 / 127.5).mean(), 1e-3)
