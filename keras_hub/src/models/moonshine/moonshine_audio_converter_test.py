import keras

# Imported for testing with ragged ops.
import tensorflow as tf

from keras_hub.src.models.moonshine.moonshine_audio_converter import (
    MoonshineAudioConverter,
)
from keras_hub.src.tests.test_case import TestCase


class MoonshineAudioConverterTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "sampling_rate": 100,  # Smaller sampling rate for testing.
            "max_audio_length": 5,
        }
        audio_tensor_1 = keras.ops.ones((2,), dtype="float32")
        audio_tensor_2 = keras.ops.ones((25,), dtype="float32")
        self.input_data = tf.ragged.stack(
            [audio_tensor_1, audio_tensor_2], axis=0
        )

    def run_preprocessing_layer_test(self, cls, init_kwargs, input_data):
        layer = cls(**init_kwargs)
        output = layer(input_data)
        shape = keras.ops.shape(output)
        shape_np = keras.ops.convert_to_numpy(shape)
        self.assertEqual(len(shape_np), 2)

    def test_feature_extractor_basics(self):
        self.run_preprocessing_layer_test(
            cls=MoonshineAudioConverter,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_correctness(self):
        audio_tensor = keras.ops.convert_to_tensor(
            [1.0, 2.0, 3.0], dtype="float32"
        )
        converter = MoonshineAudioConverter(**self.init_kwargs)
        outputs = converter(audio_tensor)
        output_shape = keras.ops.shape(outputs)
        output_shape_np = keras.ops.convert_to_numpy(output_shape)
        self.assertEqual(tuple(output_shape_np), (1, 500))
        expected_prefix = keras.ops.convert_to_tensor(
            [1.0, 2.0, 3.0], dtype="float32"
        )
        zeros_tensor = keras.ops.zeros((500 - 3,), dtype="float32")
        expected = keras.ops.concatenate(
            [expected_prefix, zeros_tensor], axis=0
        )
        self.assertAllClose(outputs[0], expected, atol=1e-5, rtol=1e-5)
