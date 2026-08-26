import numpy as np

from keras_hub.src.models.mistral.mistral_image_converter import (
    Mistral3ImageConverter,
)
from keras_hub.src.tests.test_case import TestCase


class Mistral3ImageConverterTest(TestCase):
    def setUp(self):
        self.longest_edge = 16
        self.patch_size = 4
        # Isolate resize-rounding tests from the patch-merger granularity by
        # setting `spatial_merge_size=1`, so the effective rounding multiple
        # is `patch_size` alone.
        self.spatial_merge_size = 1

    def _converter(self, **kwargs):
        kwargs.setdefault("longest_edge", self.longest_edge)
        kwargs.setdefault("patch_size", self.patch_size)
        kwargs.setdefault("spatial_merge_size", self.spatial_merge_size)
        return Mistral3ImageConverter(**kwargs)

    def test_single_image_already_patch_multiple(self):
        converter = self._converter()
        image = np.zeros((8, 8, 3), dtype="float32")
        pixel_values, image_sizes = converter([image])
        self.assertEqual(pixel_values.shape, (1, 3, 8, 8))
        self.assertAllEqual(image_sizes, np.array([[8, 8]], dtype="int32"))

    def test_odd_sized_image_rounds_up_to_patch_multiple(self):
        converter = self._converter(longest_edge=32)
        image = np.zeros((17, 17, 3), dtype="float32")
        pixel_values, image_sizes = converter([image])
        # ratio = 17 / 32 < 1, so no downscale; each dim rounds up to the
        # next multiple of `patch_size=4`: (17 - 1) // 4 + 1 = 5 -> 20.
        self.assertAllEqual(image_sizes, np.array([[20, 20]], dtype="int32"))
        self.assertEqual(pixel_values.shape, (1, 3, 20, 20))

    def test_batch_with_different_sizes_reports_true_sizes_and_pads(self):
        converter = self._converter()
        # Image A: 8x8, already a patch multiple, no downscale needed.
        image_a = np.full((8, 8, 3), 255.0, dtype="float32")
        # Image B: 12x8, already a patch multiple, no downscale needed.
        image_b = np.zeros((12, 8, 3), dtype="float32")
        pixel_values, image_sizes = converter([image_a, image_b])

        self.assertAllEqual(
            image_sizes, np.array([[8, 8], [12, 8]], dtype="int32")
        )
        # Batch-local padding to this call's max (12, 8), not a fixed
        # canvas.
        self.assertEqual(pixel_values.shape, (2, 3, 12, 8))

        pixel_values = np.array(pixel_values)
        # Image A's real content occupies rows [0, 8); rows [8, 12) are
        # zero padding.
        self.assertAllClose(
            pixel_values[0, :, 8:, :], np.zeros((3, 4, 8), dtype="float32")
        )
        # A real, all-255 pixel normalizes via the CLIP formula:
        # x * scale + offset, with scale = 1/255/std and offset = -mean/std.
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype="float32")
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype="float32")
        expected_pixel = (255.0 / 255.0 - mean) / std
        self.assertAllClose(pixel_values[0, :, 0, 0], expected_pixel, atol=1e-4)

    def test_output_is_channels_first(self):
        converter = self._converter()
        image = np.zeros((8, 12, 3), dtype="float32")
        pixel_values, _ = converter([image])
        # (num_images, num_channels, height, width).
        self.assertEqual(pixel_values.shape[1], 3)
        self.assertEqual(len(pixel_values.shape), 4)

    def test_config(self):
        converter = self._converter()
        self.assertEqual(converter.longest_edge, self.longest_edge)
        self.assertEqual(converter.patch_size, self.patch_size)
        self.assertEqual(converter.spatial_merge_size, self.spatial_merge_size)
        config = converter.get_config()
        self.assertEqual(config["longest_edge"], self.longest_edge)
        self.assertEqual(config["patch_size"], self.patch_size)
        self.assertEqual(config["spatial_merge_size"], self.spatial_merge_size)

    def test_spatial_merge_size_widens_rounding_multiple(self):
        converter = self._converter(longest_edge=32, spatial_merge_size=2)
        image = np.zeros((9, 9, 3), dtype="float32")
        pixel_values, image_sizes = converter([image])
        self.assertAllEqual(image_sizes, np.array([[16, 16]], dtype="int32"))
        self.assertEqual(pixel_values.shape, (1, 3, 16, 16))
