import numpy as np
import pytest

from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_converter import (
    Qwen3ASRAudioConverter,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen3ASRAudioConverterTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "num_mels": 128,
            "sampling_rate": 16000,
            "max_audio_length": 1,
        }

    def test_output_shape(self):
        converter = Qwen3ASRAudioConverter(**self.init_kwargs)
        # 1 second of audio at 16kHz.
        audio = np.random.randn(1, 16000).astype("float32")
        output = converter(audio)
        # Output should be (batch, num_frames, num_mels).
        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[2], 128)

    def test_single_audio(self):
        converter = Qwen3ASRAudioConverter(**self.init_kwargs)
        # Single unbatched audio.
        audio = np.random.randn(16000).astype("float32")
        output = converter(audio)
        self.assertEqual(len(output.shape), 2)
        self.assertEqual(output.shape[-1], 128)

    def test_custom_num_mels(self):
        converter = Qwen3ASRAudioConverter(
            num_mels=80, sampling_rate=16000, max_audio_length=1
        )
        audio = np.random.randn(1, 16000).astype("float32")
        output = converter(audio)
        self.assertEqual(output.shape[2], 80)

    def test_serialization(self):
        converter = Qwen3ASRAudioConverter(**self.init_kwargs)
        config = converter.get_config()
        restored = Qwen3ASRAudioConverter.from_config(config)
        self.assertEqual(restored.get_config(), config)
