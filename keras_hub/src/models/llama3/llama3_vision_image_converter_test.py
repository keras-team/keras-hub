import numpy as np

from keras_hub.src.models.llama3.llama3_vision_image_converter import (
    Llama3VisionImageConverter,
)
from keras_hub.src.tests.test_case import TestCase


class Llama3VisionImageConverterTest(TestCase):
    def test_image_converter_basics(self):
        # Manual test to verify execution without triggering XLA compilation
        converter = Llama3VisionImageConverter(
            image_size=(16, 16),
            scale=1.0 / 255.0,
        )
        input_data = np.random.randint(0, 255, (2, 32, 32, 3)).astype("float32")

        # Run Inference
        output = converter(input_data)

        # Check Shape
        self.assertEqual(output.shape, (2, 16, 16, 3))

    def test_rescaling(self):
        # Verify that pixels are scaled to [0, 1]
        converter = Llama3VisionImageConverter(
            image_size=(10, 10), scale=1.0 / 255.0, offset=0.0
        )
        # Create a pure white image (255)
        images = np.ones((2, 20, 20, 3)).astype("float32") * 255

        output = converter(images)

        # Output should be 1.0 (float)
        self.assertAllClose(output, np.ones((2, 10, 10, 3)), atol=1e-5)

    def test_serialization(self):
        # Test layer serialization using get_config/from_config
        converter = Llama3VisionImageConverter(
            image_size=(16, 16),
            scale=1.0 / 255.0,
        )

        # Get config
        config = converter.get_config()

        # Recreate from config
        restored_converter = Llama3VisionImageConverter.from_config(config)

        # Verify config was preserved
        self.assertEqual(restored_converter.image_size, (16, 16))
        self.assertEqual(restored_converter.scale, 1.0 / 255.0)

        # Verify inference matches
        input_data = np.random.randint(0, 255, (2, 32, 32, 3)).astype("float32")
        out1 = converter(input_data)
        out2 = restored_converter(input_data)
        self.assertAllClose(out1, out2)
