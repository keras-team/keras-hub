import keras
import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.gemma3.gemma3_image_converter import (
    Gemma3ImageConverter,
)
from keras_hub.src.tests.test_case import TestCase


@pytest.mark.skipif(
    keras.config.backend() != "tensorflow",
    reason="these tests are ragged and only enabled on tensorflow backend",
)
class Gemma3ImageConverterTest(TestCase):
    def test_unbatched(self):
        converter = Gemma3ImageConverter(
            image_size=(4, 4),
            scale=(1.0 / 255.0, 0.8 / 255.0, 1.2 / 255.0),
            offset=(0.2, -0.1, 0.25),
            image_max_length=3,
        )
        inputs = [np.ones((10, 10, 3))]
        padded_images, num_valid_images = converter(inputs)
        self.assertEqual(ops.shape(padded_images), (1, 3, 4, 4, 3))
        self.assertAllEqual(
            num_valid_images,
            [1],
        )

    def test_batched(self):
        converter = Gemma3ImageConverter(
            image_size=(4, 4),
            image_max_length=5,
        )
        inputs = {
            "images": [
                np.ones((2, 10, 10, 3), dtype=np.float32),
                np.ones((1, 10, 10, 3), dtype=np.float32),
                np.ones((2, 10, 10, 3), dtype=np.float32),
            ]
        }
        output = converter(inputs)
        self.assertEqual(ops.shape(output["images"]), (3, 5, 4, 4, 3))
        self.assertAllEqual(output["num_valid_images"], (2, 1, 2))
