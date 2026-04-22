"""Tests for BLIP-2 image converter."""

import numpy as np

from keras_hub.src.models.blip2.blip2_image_converter import Blip2ImageConverter
from keras_hub.src.tests.test_case import TestCase


class Blip2ImageConverterTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "image_size": (224, 224),
            "crop_to_aspect_ratio": True,
            "interpolation": "bicubic",
        }
        self.input_data = np.ones((2, 100, 100, 3), dtype="float32") * 128

    def test_image_converter_basics(self):
        kwargs = {**self.init_kwargs, "interpolation": "bilinear"}
        self.run_preprocessing_layer_test(
            cls=Blip2ImageConverter,
            init_kwargs=kwargs,
            input_data=self.input_data,
        )

    def test_normalization_values(self):
        converter = Blip2ImageConverter()
        inputs = np.ones((224, 224, 3), dtype="float32") * 255.0
        outputs = converter(inputs)
        expected = (
            255.0 * Blip2ImageConverter._SCALE[0]
            + Blip2ImageConverter._OFFSET[0]
        )
        self.assertAllClose(outputs[0, 0, 0], expected)

    def test_uint8_input(self):
        converter = Blip2ImageConverter()
        inputs = np.ones((224, 224, 3), dtype="uint8") * 255
        outputs = converter(inputs)
        expected = (
            255.0 * Blip2ImageConverter._SCALE[0]
            + Blip2ImageConverter._OFFSET[0]
        )
        self.assertAllClose(outputs[0, 0, 0], expected)

    def test_output_dtype_is_float32(self):
        converter = Blip2ImageConverter(dtype="float16")
        inputs = np.ones((2, 100, 100, 3), dtype="float32")
        outputs = converter(inputs)
        self.assertIn("float32", str(outputs.dtype))

    def test_crop_to_aspect_ratio(self):
        converter = Blip2ImageConverter(
            image_size=(224, 224), crop_to_aspect_ratio=True
        )
        inputs = np.ones((100, 400, 3), dtype="float32")
        outputs = converter(inputs)
        self.assertEqual(outputs.shape[-3:-1], (224, 224))

    def test_interpolation_methods_differ(self):
        converter_bicubic = Blip2ImageConverter(interpolation="bicubic")
        converter_nearest = Blip2ImageConverter(interpolation="nearest")
        inputs = np.random.uniform(size=(100, 100, 3)).astype("float32")
        self.assertNotAllClose(
            converter_bicubic(inputs), converter_nearest(inputs)
        )
