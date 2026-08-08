import numpy as np
from keras import ops

from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_converter import (
    Qwen3ASRAudioConverter,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen3ASRAudioConverterTest(TestCase):
    """Tests for Qwen3ASRAudioConverter."""

    def setUp(self):
        self.init_kwargs = {
            "num_mels": 8,
            "num_fft_bins": 8,
            "stride": 2,
            "sampling_rate": 100,
            "max_audio_length": 1,
            "min_frequency": 0.0,
            "max_frequency": 50.0,
        }
        # Fixed-length input matching num_samples = 100.
        self.num_samples = 100  # sampling_rate * max_audio_length
        self.num_frames = self.num_samples // 2  # stride = 2 -> 50 frames
        self.input_data = np.ones((2, self.num_samples), dtype="float32")

    def test_audio_converter_basics(self):
        """Converter initialises and produces the correct output shape."""
        converter = Qwen3ASRAudioConverter(**self.init_kwargs)
        out = converter(self.input_data)
        self.assertEqual(out.shape, (2, self.num_frames, 8))

    def test_1d_input_shape(self):
        """Single waveform (no batch dim) returns (num_frames, num_mels)."""
        converter = Qwen3ASRAudioConverter(**self.init_kwargs)
        waveform = np.ones((self.num_samples,), dtype="float32")
        out = converter(waveform)
        self.assertEqual(out.shape, (self.num_frames, 8))

    def test_2d_input_shape(self):
        """Batched waveform returns (batch_size, num_frames, num_mels)."""
        converter = Qwen3ASRAudioConverter(**self.init_kwargs)
        waveform = np.ones((3, self.num_samples), dtype="float32")
        out = converter(waveform)
        self.assertEqual(out.shape, (3, self.num_frames, 8))

    def test_short_input_is_padded(self):
        """Audio shorter than num_samples is zero-padded to the fixed length."""
        converter = Qwen3ASRAudioConverter(**self.init_kwargs)
        short = np.ones((40,), dtype="float32")  # shorter than 100
        out = converter(short)
        self.assertEqual(out.shape, (self.num_frames, 8))

    def test_long_input_is_trimmed(self):
        """Audio longer than num_samples is trimmed to the fixed length."""
        converter = Qwen3ASRAudioConverter(**self.init_kwargs)
        long = np.ones((200,), dtype="float32")  # longer than 100
        out = converter(long)
        self.assertEqual(out.shape, (self.num_frames, 8))

    def test_output_is_finite(self):
        """Log-mel outputs should be finite."""
        converter = Qwen3ASRAudioConverter(**self.init_kwargs)
        waveform = (
            np.random.default_rng(42)
            .standard_normal(self.num_samples)
            .astype("float32")
        )
        out = converter(waveform)
        self.assertFalse(bool(ops.any(ops.isnan(out))), "Output contains NaN")
        self.assertFalse(bool(ops.any(ops.isinf(out))), "Output contains Inf")

    def test_get_config_round_trip(self):
        """get_config / from_config should reproduce identical parameters."""
        converter = Qwen3ASRAudioConverter(**self.init_kwargs)
        config = converter.get_config()
        restored = Qwen3ASRAudioConverter.from_config(config)
        for key, val in self.init_kwargs.items():
            self.assertEqual(getattr(restored, key), val)

    def test_default_parameters(self):
        """Verify the default parameter values."""
        converter = Qwen3ASRAudioConverter()
        self.assertEqual(converter.num_mels, 128)
        self.assertEqual(converter.num_fft_bins, 400)
        self.assertEqual(converter.stride, 160)
        self.assertEqual(converter.sampling_rate, 16000)
        self.assertEqual(converter.max_audio_length, 30)
        self.assertAlmostEqual(converter.min_frequency, 0.0)
        self.assertAlmostEqual(converter.max_frequency, 8000.0)
        # num_samples = 16000 * 30 = 480_000
        self.assertEqual(converter.num_samples, 480_000)

    def test_default_output_shape(self):
        """1 second of 16 kHz audio padded to max_audio_length * sample_rate."""
        converter = Qwen3ASRAudioConverter()
        waveform = np.zeros(16000, dtype=np.float32)
        out = converter(waveform)
        # num_frames = (16000 * 30) // 160 = 3000
        self.assertEqual(out.shape[-1], 128)
        self.assertEqual(out.shape[-2], 3000)
