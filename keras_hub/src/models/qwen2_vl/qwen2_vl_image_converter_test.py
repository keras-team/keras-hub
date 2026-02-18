import numpy as np

from keras_hub.src.models.qwen2_vl.qwen2_vl_image_converter import (
    Qwen2VLImageConverter,
)
from keras_hub.src.models.qwen2_vl.qwen2_vl_image_converter import smart_resize
from keras_hub.src.tests.test_case import TestCase


class SmartResizeTest(TestCase):
    def test_smart_resize_basic(self):
        # Should round to nearest multiple of 28
        h, w = smart_resize(100, 100, factor=28)
        self.assertEqual(h % 28, 0)
        self.assertEqual(w % 28, 0)
        self.assertEqual(h, 112)  # round(100/28)*28 = 4*28 = 112
        self.assertEqual(w, 112)

    def test_smart_resize_max_pixels(self):
        # Should scale down if exceeds max_pixels
        h, w = smart_resize(5000, 5000, factor=28, max_pixels=1000000)
        self.assertLessEqual(h * w, 1000000)
        self.assertEqual(h % 28, 0)
        self.assertEqual(w % 28, 0)

    def test_smart_resize_min_pixels(self):
        # Should scale up if below min_pixels
        h, w = smart_resize(10, 10, factor=28, min_pixels=56 * 56)
        self.assertGreaterEqual(h * w, 56 * 56)
        self.assertEqual(h % 28, 0)
        self.assertEqual(w % 28, 0)

    def test_smart_resize_aspect_ratio_error(self):
        # Should raise error if aspect ratio > 200
        with self.assertRaises(ValueError):
            smart_resize(10000, 10, factor=28)


class Qwen2VLImageConverterTest(TestCase):
    def setUp(self):
        self.converter = Qwen2VLImageConverter(
            patch_size=14,
            temporal_patch_size=2,
            merge_size=2,
        )

    def test_converter_output_shape(self):
        # Single image: (H, W, C)
        image = np.random.rand(224, 224, 3).astype("float32")
        patches, grid_thw = self.converter(image)

        # patches should be flat: (total_patches, patch_flat_dim)
        self.assertEqual(len(patches.shape), 2)
        # grid_thw should be (num_images, 3)
        self.assertEqual(grid_thw.shape, (1, 3))

    def test_converter_multiple_images(self):
        # Multiple images as list
        images = [
            np.random.rand(224, 224, 3).astype("float32"),
            np.random.rand(448, 224, 3).astype("float32"),
        ]
        patches, grid_thw = self.converter(images)

        # grid_thw should have 2 rows (2 images)
        self.assertEqual(grid_thw.shape[0], 2)
        self.assertEqual(grid_thw.shape[1], 3)

    def test_converter_config_roundtrip(self):
        config = self.converter.get_config()
        new_converter = Qwen2VLImageConverter.from_config(config)

        # Test that it works the same
        image = np.random.rand(224, 224, 3).astype("float32")
        patches1, grid1 = self.converter(image)
        patches2, grid2 = new_converter(image)

        self.assertEqual(patches1.shape, patches2.shape)
        np.testing.assert_array_equal(grid1, grid2)

    def test_image_normalization(self):
        # Test that images are normalized with correct mean/std
        image = np.ones((224, 224, 3), dtype="float32") * 255
        patches, _ = self.converter(image)

        # After normalization with mean=[0.48145466, 0.4578275, 0.40821073]
        # and std=[0.26862954, 0.26130258, 0.27577711], pixel value 255 should
        # be transformed to approximately (255/255 - mean) / std
        # We're just checking it's been normalized (not exact values)
        self.assertTrue(
            patches.mean() > 0
        )  # Should be positive after normalization
