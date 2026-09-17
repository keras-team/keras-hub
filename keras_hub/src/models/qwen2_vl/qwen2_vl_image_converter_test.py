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
        self.assertEqual(h, 112)
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
        image = np.random.rand(224, 224, 3).astype("float32") * 255
        out = self.converter(image)

        # patches: (total_patches, temporal_patch_size, p, p, C)
        self.assertEqual(len(out["patches"].shape), 5)
        self.assertEqual(out["patches"].shape[1], 2)
        self.assertEqual(out["patches"].shape[2], 14)
        self.assertEqual(out["patches"].shape[3], 14)
        self.assertEqual(out["patches"].shape[4], 3)
        # grid_thw should be (num_images, 3)
        self.assertEqual(out["grid_thw"].shape, (1, 3))

    def test_converter_multiple_images(self):
        # Multiple images as list
        images = [
            np.random.rand(224, 224, 3).astype("float32") * 255,
            np.random.rand(448, 224, 3).astype("float32") * 255,
        ]
        out = self.converter(images)

        # grid_thw should have 2 rows (2 images)
        self.assertEqual(out["grid_thw"].shape[0], 2)
        self.assertEqual(out["grid_thw"].shape[1], 3)

    def test_converter_config_roundtrip(self):
        config = self.converter.get_config()
        new_converter = Qwen2VLImageConverter.from_config(config)

        # Test that it works the same
        image = np.random.rand(224, 224, 3).astype("float32") * 255
        out1 = self.converter(image)
        out2 = new_converter(image)

        self.assertEqual(out1["patches"].shape, out2["patches"].shape)
        np.testing.assert_array_equal(out1["grid_thw"], out2["grid_thw"])

    def test_image_normalization(self):
        # Test that images are normalized with correct mean/std.
        # A uniform 255 image becomes (1.0 - mean) / std per channel.
        image = np.ones((224, 224, 3), dtype="float32") * 255
        out = self.converter(image)
        patches = out["patches"]

        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        expected = (1.0 - mean) / std  # per-channel expected values

        # patches: (N, tps, p, p, C) — channel is the last axis.
        # Average in float64 to avoid pairwise-summation drift.
        channel_means = patches.astype("float64").mean(axis=(0, 1, 2, 3))

        np.testing.assert_allclose(
            channel_means, expected, atol=1e-5, rtol=1e-5
        )
