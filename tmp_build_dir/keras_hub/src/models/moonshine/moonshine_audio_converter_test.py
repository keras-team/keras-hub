import keras
import numpy as np

from keras_hub.src.models.moonshine.moonshine_audio_converter import (
    MoonshineAudioConverter,
)
from keras_hub.src.tests.test_case import TestCase


class MoonshineAudioConverterTest(TestCase):
    def setUp(self):
        self.sampling_rate = 16000
        self.padding_value = 0.0
        self.do_normalize = False
        self.preprocessor = MoonshineAudioConverter(
            sampling_rate=self.sampling_rate,
            padding_value=self.padding_value,
            do_normalize=self.do_normalize,
        )
        self.input_data = keras.ops.convert_to_tensor(
            [[0.1] * self.sampling_rate], dtype="float32"
        )
        self.input_data = keras.ops.expand_dims(self.input_data, axis=-1)
        self.init_kwargs = {
            "sampling_rate": self.sampling_rate,
            "padding_value": self.padding_value,
            "do_normalize": self.do_normalize,
        }

    def test_output_shape(self):
        output = self.preprocessor(self.input_data)
        self.assertEqual(keras.ops.shape(output), (1, self.sampling_rate, 1))

    def test_padding(self):
        max_length = 20000
        output = self.preprocessor(
            self.input_data, padding="max_length", max_length=max_length
        )
        self.assertEqual(keras.ops.shape(output), (1, max_length, 1))

    def test_normalization(self):
        preprocessor_no_norm = MoonshineAudioConverter(
            sampling_rate=self.sampling_rate,
            padding_value=self.padding_value,
            do_normalize=False,
        )
        preprocessor_norm = MoonshineAudioConverter(
            sampling_rate=self.sampling_rate,
            padding_value=self.padding_value,
            do_normalize=True,
        )
        input_data = keras.ops.convert_to_tensor(
            np.arange(self.sampling_rate, dtype=np.float32) / self.sampling_rate
        )  # Values from 0 to ~1
        input_data = keras.ops.expand_dims(input_data, axis=0)  # (1, 16000)
        input_data = keras.ops.expand_dims(input_data, axis=-1)  # (1, 16000, 1)
        output_no_norm = preprocessor_no_norm(input_data)
        output_norm = preprocessor_norm(input_data)
        self.assertEqual(
            keras.ops.shape(output_no_norm), keras.ops.shape(output_norm)
        )
        self.assertFalse(keras.ops.all(output_no_norm == output_norm))
        mean = keras.ops.mean(output_norm, axis=1, keepdims=True)
        var = keras.ops.var(output_norm, axis=1, keepdims=True)
        self.assertAllClose(mean, keras.ops.zeros_like(mean), atol=1e-6)
        self.assertAllClose(var, keras.ops.ones_like(var), atol=1e-6)

    def test_sampling_rate_validation(self):
        # Test with the correct sampling rate (should not raise an error).
        self.preprocessor(
            self.input_data, sampling_rate=self.preprocessor.sampling_rate
        )
        # Test with an incorrect sampling rate (should raise ValueError).
        with self.assertRaises(ValueError):
            self.preprocessor(self.input_data, sampling_rate=8000)

    def test_get_config(self):
        config = self.preprocessor.get_config()
        self.assertIsInstance(config, dict)
        self.assertEqual(config["sampling_rate"], self.sampling_rate)
        self.assertEqual(config["padding_value"], self.padding_value)
        self.assertEqual(config["do_normalize"], self.do_normalize)

    def test_correctness(self):
        audio_input = keras.ops.convert_to_tensor(
            [[1.0, 2.0, 3.0] + [0.0] * (self.sampling_rate - 3)],
            dtype="float32",
        )
        audio_input = keras.ops.expand_dims(audio_input, axis=-1)
        converter = MoonshineAudioConverter(**self.init_kwargs)

        outputs = converter(audio_input)
        self.assertEqual(keras.ops.shape(outputs), (1, self.sampling_rate, 1))

    def test_serialization(self):
        instance = MoonshineAudioConverter(**self.init_kwargs)
        self.run_serialization_test(instance=instance)
