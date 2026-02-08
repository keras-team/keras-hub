import pytest

try:
    import tensorflow as tf
except ImportError:
    tf = None

from keras_hub.src.models.qwen3_omni.qwen3_omni_audio_converter import (
    Qwen3OmniAudioConverter,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen3OmniAudioConverterTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "num_mels": 128,
            "num_fft_bins": 400,
            "stride": 160,
            "sampling_rate": 16000,
            "max_audio_length": 2,
        }

    def test_converter_output_shape(self):
        converter = Qwen3OmniAudioConverter(**self.init_kwargs)
        # 1 second of audio at 16kHz
        audio = tf.ones((16000,), dtype="float32")
        output = converter(audio)
        # Output: (time, num_mels) for rank-1 input
        self.assertEqual(output.shape[-1], 128)

    def test_converter_batch(self):
        converter = Qwen3OmniAudioConverter(**self.init_kwargs)
        # Batch of 2 audio samples
        audio = tf.ones((2, 16000), dtype="float32")
        output = converter(audio)
        # Output: (batch, time, num_mels)
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[-1], 128)

    def test_config_serialization(self):
        converter = Qwen3OmniAudioConverter(**self.init_kwargs)
        config = converter.get_config()
        self.assertEqual(config["num_mels"], 128)
        self.assertEqual(config["num_fft_bins"], 400)
        self.assertEqual(config["stride"], 160)
        self.assertEqual(config["sampling_rate"], 16000)
        self.assertEqual(config["max_audio_length"], 2)

    def test_default_values(self):
        converter = Qwen3OmniAudioConverter()
        self.assertEqual(converter.num_mels, 128)
        self.assertEqual(converter.num_fft_bins, 400)
        self.assertEqual(converter.stride, 160)
        self.assertEqual(converter.sampling_rate, 16000)
        self.assertEqual(converter.max_audio_length, 300)

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Qwen3OmniAudioConverter.presets:
            self.run_preset_test(
                cls=Qwen3OmniAudioConverter,
                preset=preset,
                input_data=tf.ones((8000,), dtype="float32"),
            )
