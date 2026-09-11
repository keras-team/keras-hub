import numpy as np

from keras_hub.src.models.mistral3.mistral3_image_converter import (
    Mistral3ImageConverter,
)
from keras_hub.src.tests.test_case import TestCase


class Mistral3ImageConverterTest(TestCase):
    def setUp(self):
        # `spatial_merge_size=1` so the rounding multiple is `patch_size`
        # alone, isolating resize-rounding tests from patch-merger granularity.
        self.init_kwargs = {
            "longest_edge": 16,
            "patch_size": 4,
            "spatial_merge_size": 1,
        }
        self.converter = Mistral3ImageConverter(**self.init_kwargs)

    def test_image_converter_basics(self):
        image_a = np.full((8, 8, 3), 255.0, dtype="float32")
        image_b = np.zeros((8, 8, 3), dtype="float32")
        input_data = np.stack([image_a, image_b], axis=0)
        self.run_preprocessing_layer_test(
            cls=Mistral3ImageConverter,
            init_kwargs=self.init_kwargs,
            input_data=input_data,
        )

    def test_single_image_already_patch_multiple(self):
        image = np.zeros((8, 8, 3), dtype="float32")
        pixel_values, image_sizes = self.converter([image])
        self.assertEqual(pixel_values.shape, (1, 3, 8, 8))
        self.assertAllEqual(image_sizes, np.array([[8, 8]], dtype="int32"))

    def test_odd_sized_image_rounds_up_to_patch_multiple(self):
        converter = Mistral3ImageConverter(
            **{**self.init_kwargs, "longest_edge": 32}
        )
        image = np.zeros((17, 17, 3), dtype="float32")
        pixel_values, image_sizes = converter([image])
        # ratio = 17 / 32 < 1, so no downscale; each dim rounds up to the
        # next multiple of `patch_size=4`: (17 - 1) // 4 + 1 = 5 -> 20.
        self.assertAllEqual(image_sizes, np.array([[20, 20]], dtype="int32"))
        self.assertEqual(pixel_values.shape, (1, 3, 20, 20))

    def test_batch_with_different_sizes_reports_true_sizes_and_pads(self):
        image_a = np.full((8, 8, 3), 255.0, dtype="float32")
        image_b = np.zeros((12, 8, 3), dtype="float32")
        pixel_values, image_sizes = self.converter([image_a, image_b])

        self.assertAllEqual(
            image_sizes, np.array([[8, 8], [12, 8]], dtype="int32")
        )
        # Batch-local padding to this call's max (12, 8), not a fixed
        # canvas.
        self.assertEqual(pixel_values.shape, (2, 3, 12, 8))

        pixel_values = np.array(pixel_values)
        # Rows [8, 12) are zero padding added to reach the batch max height.
        self.assertAllClose(
            pixel_values[0, :, 8:, :], np.zeros((3, 4, 8), dtype="float32")
        )
        # CLIP normalization: x * scale + offset, scale = 1/255/std,
        # offset = -mean/std.
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype="float32")
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype="float32")
        expected_pixel = (255.0 / 255.0 - mean) / std
        self.assertAllClose(pixel_values[0, :, 0, 0], expected_pixel, atol=1e-4)

    def test_spatial_merge_size_widens_rounding_multiple(self):
        converter = Mistral3ImageConverter(
            **{**self.init_kwargs, "longest_edge": 32, "spatial_merge_size": 2}
        )
        image = np.zeros((9, 9, 3), dtype="float32")
        pixel_values, image_sizes = converter([image])
        self.assertAllEqual(image_sizes, np.array([[16, 16]], dtype="int32"))
        self.assertEqual(pixel_values.shape, (1, 3, 16, 16))
