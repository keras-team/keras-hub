import keras

from keras_hub.src.models.moonshine.moonshine_audio_converter import (
    MoonshineAudioConverter,
)
from keras_hub.src.tests.test_case import TestCase


class MoonshineAudioConverterTest(TestCase):
    def setUp(self):
        super().setUp()
        self.filter_dim = 256
        self.preprocessor = MoonshineAudioConverter(filter_dim=self.filter_dim)
        self.input_data = keras.ops.convert_to_tensor(
            [[0.1] * 16000], dtype="float32"
        )
        self.input_data = keras.ops.expand_dims(self.input_data, axis=-1)
        self.init_kwargs = {
            "filter_dim": self.filter_dim,
            "sampling_rate": 16000,
            "padding_value": 0.0,
            "do_normalize": False,
            "return_attention_mask": True,
            "initializer_range": 0.02,
        }

    def test_output_shape(self):
        output = self.preprocessor(self.input_data)
        self.assertEqual(
            keras.ops.shape(output["input_values"]), (1, 40, self.filter_dim)
        )
        self.assertEqual(keras.ops.shape(output["attention_mask"]), (1, 40))
        self.assertAllEqual(
            output["attention_mask"], keras.ops.ones((1, 40), dtype="int32")
        )

    def test_padding(self):
        max_length = 20000
        output = self.preprocessor(
            self.input_data, padding="max_length", max_length=max_length
        )
        self.assertEqual(
            keras.ops.shape(output["input_values"]), (1, 50, self.filter_dim)
        )
        self.assertEqual(keras.ops.shape(output["attention_mask"]), (1, 50))
        expected_mask = keras.ops.concatenate(
            [
                keras.ops.ones((1, 40), dtype="int32"),
                keras.ops.zeros((1, 10), dtype="int32"),
            ],
            axis=1,
        )
        self.assertAllEqual(output["attention_mask"], expected_mask)

    def test_normalization(self):
        preprocessor_no_norm = MoonshineAudioConverter(
            filter_dim=self.filter_dim, do_normalize=False
        )
        preprocessor_norm = MoonshineAudioConverter(
            filter_dim=self.filter_dim, do_normalize=True
        )
        input_data = keras.ops.arange(16000, dtype="float32") / 16000  # Values
        # from 0 to ~1
        input_data = keras.ops.expand_dims(input_data, axis=0)  # (1, 16000)
        input_data = keras.ops.expand_dims(input_data, axis=-1)  # (1, 16000, 1)
        output_no_norm = preprocessor_no_norm(input_data)
        output_norm = preprocessor_norm(input_data)
        self.assertFalse(
            keras.ops.all(
                output_no_norm["input_values"] == output_norm["input_values"]
            )
        )

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
        self.assertEqual(config["filter_dim"], self.filter_dim)
        self.assertEqual(config["sampling_rate"], 16000)
        self.assertEqual(config["padding_value"], 0.0)
        self.assertEqual(config["do_normalize"], False)
        self.assertEqual(config["return_attention_mask"], True)
        self.assertEqual(config["initializer_range"], 0.02)

    def test_correctness(self):
        audio_input = keras.ops.convert_to_tensor(
            [[1.0, 2.0, 3.0] + [0.0] * 15997], dtype="float32"
        )
        audio_input = keras.ops.expand_dims(audio_input, axis=-1)
        converter = MoonshineAudioConverter(**self.init_kwargs)

        outputs = converter(audio_input)
        self.assertIn("input_values", outputs)
        self.assertIn("attention_mask", outputs)

        self.assertEqual(
            keras.ops.shape(outputs["input_values"]), (1, 40, self.filter_dim)
        )
        self.assertEqual(keras.ops.shape(outputs["attention_mask"]), (1, 40))
        self.assertAllEqual(
            outputs["attention_mask"], keras.ops.ones((1, 40), dtype="int32")
        )

    def test_serialization(self):
        instance = MoonshineAudioConverter(**self.init_kwargs)
        self.run_serialization_test(instance=instance)
