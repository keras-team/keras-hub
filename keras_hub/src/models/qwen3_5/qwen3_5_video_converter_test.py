import pytest
from keras import ops

from keras_hub.src.models.qwen3_5.qwen3_5_video_converter import (
    Qwen3_5VideoConverter,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen3_5VideoConverterTest(TestCase):
    def setUp(self):
        self.converter = Qwen3_5VideoConverter(
            patch_size=14,
            temporal_patch_size=2,
            spatial_merge_size=2,
            min_pixels=1000,
            max_pixels=50000,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
        )

    def test_call_ops(self):
        """Even temporal size, within pixel budget."""
        inputs = ops.ones((4, 60, 60, 3), dtype="float32")
        outputs = self.converter(inputs)
        self.assertEqual(outputs["patches"].shape, (32, 2, 14, 14, 3))
        self.assertAllEqual(outputs["grid_thw"], [2, 4, 4])

    def test_call_ops_padding(self):
        """Odd temporal size -> requires padding."""
        inputs = ops.ones((3, 60, 60, 3), dtype="float32")
        outputs = self.converter(inputs)
        self.assertEqual(outputs["patches"].shape, (32, 2, 14, 14, 3))
        self.assertAllEqual(outputs["grid_thw"], [2, 4, 4])

    def test_call_ops_downscaling(self):
        """Needs downscaling due to temporal * spatial exceeding max_pixels."""
        inputs = ops.ones((10, 200, 200, 3), dtype="float32")
        outputs = self.converter(inputs)
        self.assertEqual(outputs["patches"].shape, (80, 2, 14, 14, 3))
        self.assertAllEqual(outputs["grid_thw"], [5, 4, 4])

    def test_call_ops_upscaling(self):
        """Very small frames, no upscaling (above min_pixels)."""
        inputs = ops.ones((2, 20, 20, 3), dtype="float32")
        outputs = self.converter(inputs)
        self.assertEqual(outputs["patches"].shape, (4, 2, 14, 14, 3))
        self.assertAllEqual(outputs["grid_thw"], [1, 2, 2])

    @pytest.mark.extra_large
    def test_tf_dataset_map(self):
        import tensorflow as tf

        ds = tf.data.Dataset.from_tensor_slices(
            [ops.ones((4, 60, 60, 3), dtype="float32")]
        )
        ds = ds.map(self.converter)
        outputs = next(iter(ds))
        self.assertEqual(outputs["patches"].shape, (32, 2, 14, 14, 3))
        self.assertAllEqual(outputs["grid_thw"], [2, 4, 4])
