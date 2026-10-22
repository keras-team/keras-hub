import grain
import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.qwen3_vl.qwen3_vl_image_converter import (
    Qwen3_VLImageConverter,
)
from keras_hub.src.models.qwen3_vl.qwen3_vl_image_converter import smart_resize
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.tests.test_case import assert_grain_safe_types

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import torch
    import torchvision  # noqa: F401
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        Qwen2VLImageProcessor,
    )
except ImportError:
    torch = None


def _smooth_image(height, width):
    ys, xs = np.mgrid[:height, :width]
    channels = [
        40 + 150 * ys / height,
        40 + 150 * xs / width,
        40 + 75 * (ys / height + xs / width),
    ]
    return np.stack(channels, axis=-1).astype("float32")


class Qwen3_VLImageConverterTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "patch_size": 4,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "min_pixels": 1024,
            "max_pixels": 4096,
        }
        self.converter = Qwen3_VLImageConverter(**self.init_kwargs)

    def test_defaults(self):
        converter = Qwen3_VLImageConverter()
        self.assertEqual(converter.interpolation, "bicubic")
        self.assertAllClose(converter.scale, 2.0 / 255.0)
        self.assertAllClose(converter.offset, -1.0)
        self.assertEqual(converter.min_pixels, 65536)
        self.assertEqual(converter.max_pixels, 16777216)

    def test_smart_resize(self):
        # Inside the budget: rounded to the factor.
        self.assertEqual(smart_resize(70, 50, 8, 1024, 4096), (72, 48))
        # Above the budget: shrunk, rounded down.
        self.assertEqual(smart_resize(200, 100, 8, 1024, 4096), (88, 40))
        # Below the budget: grown, rounded up.
        self.assertEqual(smart_resize(16, 16, 8, 1024, 4096), (32, 32))
        with self.assertRaises(ValueError):
            smart_resize(10, 3000, 8, 1024, 4096)

    def test_output_shapes_and_grid(self):
        outputs = self.converter(np.ones((48, 64, 3), "float32") * 100)
        self.assertEqual(outputs["patches"].shape, (12 * 16 // 1, 2, 4, 4, 3))
        self.assertAllEqual(outputs["grid_thw"], [1, 12, 16])

    def test_resized_grid(self):
        outputs = self.converter(np.ones((200, 100, 3), "uint8"))
        self.assertAllEqual(outputs["grid_thw"], [1, 22, 10])
        self.assertEqual(outputs["patches"].shape, (220, 2, 4, 4, 3))

    def test_temporal_duplication_and_normalization(self):
        outputs = self.converter(np.full((48, 64, 3), 255, "uint8"))
        self.assertAllClose(outputs["patches"], np.ones((192, 2, 4, 4, 3)))
        outputs = self.converter(np.zeros((48, 64, 3), "uint8"))
        self.assertAllClose(outputs["patches"], -np.ones((192, 2, 4, 4, 3)))

    def test_merge_block_order(self):
        # Without resizing, each patch holds its raster index as its value.
        converter = Qwen3_VLImageConverter(
            patch_size=1,
            spatial_merge_size=2,
            min_pixels=1,
            max_pixels=1000,
            scale=1.0,
            offset=0.0,
        )
        image = np.arange(16, dtype="float32").reshape(4, 4, 1)
        image = np.repeat(image, 3, axis=-1)
        patches = ops.convert_to_numpy(converter(image)["patches"])
        self.assertAllEqual(
            patches[:, 0, 0, 0, 0],
            [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15],
        )

    def test_get_config_round_trip(self):
        self.run_serialization_test(self.converter)
        config = self.converter.get_config()
        for key, value in self.init_kwargs.items():
            self.assertEqual(config[key], value)

    def test_grain_map(self):
        image = np.ones((200, 100, 3), "uint8") * 7
        expected = self.converter(image)
        ds = grain.MapDataset.source([image, image]).map(self.converter)
        for element in ds:
            assert_grain_safe_types(element)
            self.assertAllClose(element["patches"], expected["patches"])
            self.assertAllEqual(element["grid_thw"], [1, 22, 10])

    @pytest.mark.skipif(tf is None, reason="Requires tensorflow.")
    def test_tf_dataset_map(self):
        image = np.random.RandomState(0).randint(
            0, 256, (200, 100, 3), dtype="uint8"
        )
        ds = tf.data.Dataset.from_tensor_slices([image])
        outputs = next(iter(ds.map(self.converter)))
        eager = self.converter(image)
        self.assertAllEqual(outputs["grid_thw"], [1, 22, 10])
        self.assertEqual(outputs["patches"].shape, (220, 2, 4, 4, 3))
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
    def test_matches_hf_image_processor(self):
        hf = Qwen2VLImageProcessor(
            size={"shortest_edge": 1024, "longest_edge": 4096},
            patch_size=4,
            temporal_patch_size=2,
            merge_size=2,
            image_mean=[0.5] * 3,
            image_std=[0.5] * 3,
        )
        # Aligned (no resize), downscaled, and upscaled inputs.
        for height, width in [(48, 64), (200, 100), (20, 36)]:
            image = _smooth_image(height, width)
            expected = hf(
                images=torch.from_numpy(image).permute(2, 0, 1),
                return_tensors="np",
            )
            outputs = self.converter(image)
            # HF flattens each patch as (channels, time, rows, cols).
            n = expected["pixel_values"].shape[0]
            hf_patches = expected["pixel_values"].reshape(n, 3, 2, 4, 4)
            hf_patches = hf_patches.transpose(0, 2, 3, 4, 1)
            self.assertAllEqual(
                outputs["grid_thw"], expected["image_grid_thw"][0]
            )
            self.assertAllClose(
                outputs["patches"], hf_patches, atol=1e-4, rtol=0
            )

    @pytest.mark.skipif(
        torch is None, reason="Requires torch, torchvision and transformers."
    )
    def test_uint8_matches_hf_image_processor(self):
        hf = Qwen2VLImageProcessor(
            size={"shortest_edge": 1024, "longest_edge": 4096},
            patch_size=4,
            temporal_patch_size=2,
            merge_size=2,
            image_mean=[0.5] * 3,
            image_std=[0.5] * 3,
        )
        rng = np.random.default_rng(0)
        # HF resizes uint8 images in integer passes. Textured inputs make a
        # single float resize drift by a pixel level on many values.
        for height, width in [(200, 100), (130, 170), (20, 36)]:
            image = rng.integers(0, 256, (height, width, 3)).astype("uint8")
            expected = hf(
                images=torch.from_numpy(image).permute(2, 0, 1),
                return_tensors="np",
            )
            outputs = self.converter(image)
            n = expected["pixel_values"].shape[0]
            hf_patches = expected["pixel_values"].reshape(n, 3, 2, 4, 4)
            hf_patches = hf_patches.transpose(0, 2, 3, 4, 1)
            diff = np.abs(ops.convert_to_numpy(outputs["patches"]) - hf_patches)
            self.assertLess(diff.mean(), 2e-3)
            self.assertLess((diff > 2.0 / 127.5).mean(), 1e-3)
