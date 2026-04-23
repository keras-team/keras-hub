import numpy as np
import tensorflow as tf
from keras import ops

from keras_hub.src.models.smolvlm2.smolvlm2_image_converter import (
    SmolVLM2ImageConverter,
)


class SmolVLM2ImageConverterTest(tf.test.TestCase):
    def test_single_crop_output(self):
        """Small image that fits in one crop produces 1 sub-image."""
        converter = SmolVLM2ImageConverter(
            max_image_size=32,
            size=32,
            do_image_splitting=True,
            scale=[1 / 255.0] * 3,
            offset=[0.0] * 3,
        )
        # 10x10 image — after resize to longest_edge=32 it stays <=32,
        # so it fits in a single 32×32 crop.
        img = np.random.randint(0, 256, size=(10, 10, 3)).astype("uint8")
        result = converter(img)

        self.assertIn("pixel_values", result)
        self.assertIn("rows", result)
        self.assertIn("cols", result)

        pixel_values = ops.convert_to_numpy(result["pixel_values"])
        # Single crop → rows=0, cols=0.
        self.assertEqual(int(result["rows"]), 0)
        self.assertEqual(int(result["cols"]), 0)
        # Shape: (1, max_image_size, max_image_size, 3)
        self.assertEqual(pixel_values.shape, (1, 32, 32, 3))

    def test_multi_crop_output(self):
        """Larger image is split into multiple crops + global view."""
        converter = SmolVLM2ImageConverter(
            max_image_size=16,
            size=64,
            do_image_splitting=True,
            scale=[1 / 255.0] * 3,
            offset=[0.0] * 3,
        )
        # 100x100 image — after resize to longest_edge=64 and snap to
        # multiples of 16, should produce multiple 16×16 crops.
        img = np.random.randint(0, 256, size=(100, 100, 3)).astype("uint8")
        result = converter(img)

        pixel_values = ops.convert_to_numpy(result["pixel_values"])
        num_rows = int(result["rows"])
        num_cols = int(result["cols"])

        # Should be split.
        self.assertGreater(num_rows, 0)
        self.assertGreater(num_cols, 0)

        # num_sub_images = rows * cols + 1 (global view).
        expected_sub_images = num_rows * num_cols + 1
        self.assertEqual(pixel_values.shape[0], expected_sub_images)
        self.assertEqual(pixel_values.shape[1], 16)
        self.assertEqual(pixel_values.shape[2], 16)
        self.assertEqual(pixel_values.shape[3], 3)

    def test_no_splitting(self):
        """do_image_splitting=False always produces 1 crop."""
        converter = SmolVLM2ImageConverter(
            max_image_size=16,
            size=64,
            do_image_splitting=False,
            scale=[1 / 255.0] * 3,
            offset=[0.0] * 3,
        )
        img = np.random.randint(0, 256, size=(100, 60, 3)).astype("uint8")
        result = converter(img)

        pixel_values = ops.convert_to_numpy(result["pixel_values"])
        self.assertEqual(int(result["rows"]), 0)
        self.assertEqual(int(result["cols"]), 0)
        self.assertEqual(pixel_values.shape, (1, 16, 16, 3))

    def test_normalization_range(self):
        """Verify pixels normalized to [-1, 1] with default HF params."""
        # HF uses mean=0.5, std=0.5 → scale=1/(255*0.5), offset=-1.0
        converter = SmolVLM2ImageConverter(
            max_image_size=32,
            size=64,
            do_image_splitting=False,
            scale=[1.0 / (0.5 * 255)] * 3,
            offset=[-0.5 / 0.5] * 3,
        )
        img = np.random.randint(0, 256, size=(20, 20, 3)).astype("uint8")
        result = converter(img)

        pixel_values = ops.convert_to_numpy(result["pixel_values"])
        # Should be in [-1, 1] range (approximately).
        self.assertAllInRange(pixel_values, -1.05, 1.05)

    def test_config_roundtrip(self):
        """Test serialization / deserialization."""
        converter = SmolVLM2ImageConverter(
            max_image_size=512,
            size=2048,
            do_image_splitting=True,
        )
        config = converter.get_config()
        self.assertEqual(config["max_image_size"], 512)
        self.assertEqual(config["size"], 2048)
        self.assertTrue(config["do_image_splitting"])

    def test_square_image(self):
        """Square image is handled correctly."""
        converter = SmolVLM2ImageConverter(
            max_image_size=32,
            size=64,
            do_image_splitting=True,
            scale=[1 / 255.0] * 3,
            offset=[0.0] * 3,
        )
        img = np.random.randint(0, 256, size=(32, 32, 3)).astype("uint8")
        result = converter(img)

        pixel_values = ops.convert_to_numpy(result["pixel_values"])
        self.assertEqual(pixel_values.shape[1], 32)
        self.assertEqual(pixel_values.shape[2], 32)


if __name__ == "__main__":
    tf.test.main()
