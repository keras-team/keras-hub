import numpy as np
from keras import ops

from keras_hub.src.models.gemma4.gemma4_unified_audio_converter import (
    Gemma4UnifiedAudioConverter,
)
from keras_hub.src.tests.test_case import TestCase


class Gemma4UnifiedAudioConverterTest(TestCase):
    """Tests for Gemma4UnifiedAudioConverter.

    Small parameters so tests run quickly:
        audio_samples_per_token = 10
        sampling_rate            = 100 Hz
    """

    def setUp(self):
        self.init_kwargs = {
            "audio_samples_per_token": 10,
            "sampling_rate": 100,
        }

    def test_1d_input_shape(self):
        """Single waveform (no batch dim) returns (num_tokens, feat)."""
        converter = Gemma4UnifiedAudioConverter(**self.init_kwargs)
        waveform = np.ones((100,), dtype="float32")
        out = converter(waveform)
        # 100 samples / 10 samples_per_token = 10 tokens
        self.assertEqual(out.shape, (10, 10))

    def test_2d_input_shape(self):
        """Batched waveform returns (batch, num_tokens, feat)."""
        converter = Gemma4UnifiedAudioConverter(**self.init_kwargs)
        waveform = np.ones((3, 100), dtype="float32")
        out = converter(waveform)
        self.assertEqual(out.shape, (3, 10, 10))

    def test_non_divisible_length_is_padded(self):
        """Waveform not divisible by samples_per_token is zero-padded."""
        converter = Gemma4UnifiedAudioConverter(**self.init_kwargs)
        waveform = np.ones((15,), dtype="float32")
        out = converter(waveform)
        # ceil(15 / 10) = 2 tokens
        self.assertEqual(out.shape, (2, 10))
        # Last 5 samples of second token should be zero-padded.
        out_np = ops.convert_to_numpy(out)
        np.testing.assert_array_equal(out_np[1, 5:], 0.0)

    def test_exact_divisible_length(self):
        """Waveform exactly divisible produces no padding."""
        converter = Gemma4UnifiedAudioConverter(**self.init_kwargs)
        waveform = np.arange(20, dtype="float32")
        out = ops.convert_to_numpy(converter(waveform))
        self.assertEqual(out.shape, (2, 10))
        # First token = [0..9], second = [10..19]
        np.testing.assert_array_equal(out[0], np.arange(10, dtype="float32"))
        np.testing.assert_array_equal(
            out[1], np.arange(10, 20, dtype="float32")
        )

    def test_output_preserves_values(self):
        """Raw waveform values are preserved without transformation."""
        converter = Gemma4UnifiedAudioConverter(**self.init_kwargs)
        waveform = np.arange(10, dtype="float32")
        out = ops.convert_to_numpy(converter(waveform))
        np.testing.assert_array_equal(out[0], waveform)

    def test_output_is_finite(self):
        """Output should always be finite."""
        converter = Gemma4UnifiedAudioConverter(**self.init_kwargs)
        waveform = (
            np.random.default_rng(42).standard_normal(100).astype("float32")
        )
        out = converter(waveform)
        self.assertFalse(bool(ops.any(ops.isnan(out))), "Output contains NaN")
        self.assertFalse(bool(ops.any(ops.isinf(out))), "Output contains Inf")

    def test_stride_property(self):
        """stride should equal audio_samples_per_token."""
        converter = Gemma4UnifiedAudioConverter(**self.init_kwargs)
        self.assertEqual(converter.stride, 10)

    def test_audio_subsampling_factor(self):
        """Unified converter has subsampling factor of 1."""
        converter = Gemma4UnifiedAudioConverter(**self.init_kwargs)
        self.assertEqual(converter.audio_subsampling_factor, 1)

    def test_get_config_round_trip(self):
        """get_config / from_config should reproduce identical parameters."""
        converter = Gemma4UnifiedAudioConverter(**self.init_kwargs)
        config = converter.get_config()
        restored = Gemma4UnifiedAudioConverter.from_config(config)
        for key, val in self.init_kwargs.items():
            self.assertEqual(getattr(restored, key), val)
        self.assertEqual(
            restored.stride, self.init_kwargs["audio_samples_per_token"]
        )

    def test_config_round_trip_output(self):
        """Restored converter produces identical output."""
        converter = Gemma4UnifiedAudioConverter(**self.init_kwargs)
        config = converter.get_config()
        restored = Gemma4UnifiedAudioConverter.from_config(config)
        waveform = np.ones((100,), dtype="float32")
        self.assertAllClose(
            ops.convert_to_numpy(converter(waveform)),
            ops.convert_to_numpy(restored(waveform)),
            atol=1e-6,
        )

    def test_default_parameters(self):
        """Verify the default parameter values."""
        converter = Gemma4UnifiedAudioConverter()
        self.assertEqual(converter.audio_samples_per_token, 640)
        self.assertEqual(converter.sampling_rate, 16000)
        self.assertEqual(converter.stride, 640)
        self.assertEqual(converter.audio_subsampling_factor, 1)

    def test_default_output_shape(self):
        """1 second of 16 kHz audio → 25 tokens of 640 features."""
        converter = Gemma4UnifiedAudioConverter()
        waveform = np.zeros(16000, dtype="float32")
        out = converter(waveform)
        self.assertEqual(out.shape, (25, 640))
