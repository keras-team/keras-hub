import tensorflow as tf

from keras_hub.src.models.whisper.whisper_audio_converter import (
    WhisperAudioConverter,
)
from keras_hub.src.tests.test_case import TestCase


class WhisperAudioConverterTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "num_mels": 80,
            "num_fft_bins": 400,
            "stride": 100,
            "sampling_rate": 100,
            "max_audio_length": 5,
        }
        audio_tensor_1 = tf.ones((2,), dtype="float32")
        audio_tensor_2 = tf.ones((25,), dtype="float32")
        self.input_data = tf.ragged.stack(
            [audio_tensor_1, audio_tensor_2],
            axis=0,
        )

    def test_feature_extractor_basics(self):
        self.run_preprocessing_layer_test(
            cls=WhisperAudioConverter,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_correctness(self):
        audio_tensor = tf.ones((2,), dtype="float32")
        outputs = WhisperAudioConverter(**self.init_kwargs)(audio_tensor)

        # Verify shape.
        self.assertEqual(outputs.shape, (5, 80))
        # Verify output.
        expected = [1.1656, 1.0151, -0.8343, -0.8343, -0.8343]
        self.assertAllClose(outputs[:, 0], expected, atol=0.01, rtol=0.01)
