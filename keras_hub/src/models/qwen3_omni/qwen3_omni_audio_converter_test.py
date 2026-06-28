import numpy as np
import pytest

from keras_hub.src.models.qwen3_omni.qwen3_omni_audio_converter import (
    Qwen3OmniAudioConverter,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen3OmniAudioConverterTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "num_mels": 128,
            "num_fft_bins": 400,
            "stride": 100,
            "sampling_rate": 100,
            "max_audio_length": 5,
        }

    def test_converter_output_shape(self):
        converter = Qwen3OmniAudioConverter(**self.init_kwargs)
        audio = np.ones((2,), dtype="float32")
        output = converter(audio)
        self.assertEqual(output.shape[-1], 128)

    def test_converter_batch(self):
        converter = Qwen3OmniAudioConverter(**self.init_kwargs)
        audio = np.ones((2, 25), dtype="float32")
        output = converter(audio)
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[-1], 128)

    def test_config_serialization(self):
        converter = Qwen3OmniAudioConverter(**self.init_kwargs)
        config = converter.get_config()
        self.assertEqual(config["num_mels"], 128)
        self.assertEqual(config["num_fft_bins"], 400)
        self.assertEqual(config["stride"], 100)
        self.assertEqual(config["sampling_rate"], 100)
        self.assertEqual(config["max_audio_length"], 5)

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
                input_data=np.ones((800,), dtype="float32"),
            )
