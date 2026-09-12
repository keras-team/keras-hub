import numpy as np

from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_converter import (
    Qwen3ASRAudioConverter,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen3ASRAudioConverterTest(TestCase):
    def test_converter_call(self):
        # 1 second of audio at 16000Hz
        audio = np.random.uniform(size=(16000,))

        converter = Qwen3ASRAudioConverter(
            num_mels=128,
            sampling_rate=16000,
            max_audio_length=30,  # default 30s
            n_window=50,
        )

        # Call with unbatched input
        output = converter(audio)
        # Expected shape: (padded_frames, num_mels)
        # padded_frames should be a multiple of 100.
        # For max_audio_length=30, num_samples = 480000.
        # num_frames = 480000 // 160 = 3000.
        # 3000 is a multiple of 100.
        # So output shape should be (3000, 128)
        self.assertEqual(output.shape, (3000, 128))

        # Call with batched input
        batched_audio = np.random.uniform(size=(2, 16000))
        output_batched = converter(batched_audio)
        self.assertEqual(output_batched.shape, (2, 3000, 128))

    def test_audio_shape(self):
        converter = Qwen3ASRAudioConverter(
            max_audio_length=30,
            n_window=50,
        )
        self.assertEqual(converter.audio_shape(), (3000, 128))

        # Test with custom max_audio_length not a multiple of 100 frames.
        # e.g., max_audio_length = 1.05s -> 16800 samples.
        # stride = 160 -> 16800 // 160 = 105 frames.
        # Padded to multiple of 100 -> 200 frames.
        converter_short = Qwen3ASRAudioConverter(
            max_audio_length=1.05,
            n_window=50,
        )
        # Wait, max_audio_length=1.05 -> num_samples = 1.05 * 16000 = 16800.
        # 16800 // 160 = 105.
        # multiple = 100.
        # remainder = 105 % 100 = 5.
        # pad = 100 - 5 = 95.
        # padded_frames = 105 + 95 = 200.
        self.assertEqual(converter_short.audio_shape(), (200, 128))
