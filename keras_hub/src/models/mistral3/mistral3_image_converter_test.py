import numpy as np
import pytest

from keras_hub.src.models.mistral3.mistral3_image_converter import (
    Mistral3ImageConverter,
)
from keras_hub.src.models.mistral3.mistral3_image_converter import (
    _compute_resize_size_tf,
)
from keras_hub.src.models.mistral3.mistral3_image_converter import (
    compute_resize_size,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.tensor_utils import tf


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

    @pytest.mark.skipif(tf is None, reason="TensorFlow is not installed.")
    def test_tf_data_batched_unknown_height_and_width(self):
        images = np.zeros((2, 9, 6, 3), dtype="float32")
        ds = tf.data.Dataset.from_generator(
            lambda: iter([images]),
            output_signature=tf.TensorSpec((None, None, None, 3), "float32"),
        )
        pixel_values, image_sizes = next(iter(ds.map(self.converter)))
        expected_pixel_values, expected_sizes = self.converter(images)
        self.assertAllEqual(image_sizes, expected_sizes)
        self.assertAllClose(pixel_values, expected_pixel_values)

    @pytest.mark.skipif(tf is None, reason="TensorFlow is not installed.")
    def test_tf_data_unbatched_unknown_height_and_width(self):
        image = np.zeros((9, 6, 3), dtype="float32")
        ds = tf.data.Dataset.from_generator(
            lambda: iter([image]),
            output_signature=tf.TensorSpec((None, None, 3), "float32"),
        )
        pixel_values, image_sizes = next(iter(ds.map(self.converter)))
        self.assertAllEqual(image_sizes, [12, 8])
        self.assertEqual(pixel_values.shape, (3, 12, 8))

    def test_compute_resize_size_exact_multiple(self):
        # Already a multiple of `patch_size`, well under `longest_edge`:
        # no scaling, no rounding.
        size = compute_resize_size(
            height=16, width=16, longest_edge=32, patch_size=4
        )
        self.assertEqual(size, (16, 16))

    def test_compute_resize_size_rounds_up(self):
        # Under `longest_edge`, but not a `patch_size` multiple: each dim
        # rounds up independently, (17 - 1) // 4 + 1 = 5 -> 20.
        size = compute_resize_size(
            height=17, width=17, longest_edge=32, patch_size=4
        )
        self.assertEqual(size, (20, 20))

    def test_compute_resize_size_clamps_wide_image(self):
        # Wide image: width is the longest edge, so it is scaled down to
        # `longest_edge` and height is scaled by the same ratio, preserving
        # aspect ratio, before rounding up to a `patch_size` multiple.
        # ratio = 40 / 16 = 2.5 -> height = floor(20 / 2.5) = 8,
        # width = floor(40 / 2.5) = 16 (both already patch multiples).
        size = compute_resize_size(
            height=20, width=40, longest_edge=16, patch_size=4
        )
        self.assertEqual(size, (8, 16))

    def test_compute_resize_size_clamps_tall_image(self):
        # Tall image: height is the longest edge. Same scale factor is
        # applied to both dimensions.
        # ratio = 40 / 16 = 2.5 -> height = floor(40 / 2.5) = 16,
        # width = floor(20 / 2.5) = 8.
        size = compute_resize_size(
            height=40, width=20, longest_edge=16, patch_size=4
        )
        self.assertEqual(size, (16, 8))

    @pytest.mark.skipif(tf is None, reason="TensorFlow is not installed.")
    def test_compute_resize_size_tf_matches_python(self):
        for height, width in [(8, 8), (17, 5), (100, 33), (33, 100), (1, 1)]:
            expected = compute_resize_size(height, width, 16, 4)
            actual = _compute_resize_size_tf(
                tf.constant(height), tf.constant(width), 16, 4
            )
            self.assertEqual(tuple(int(v) for v in actual), expected)
