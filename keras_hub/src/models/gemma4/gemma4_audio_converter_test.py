import numpy as np
from keras import ops

from keras_hub.src.models.gemma4.gemma4_audio_converter import (
    Gemma4AudioConverter,
)
from keras_hub.src.tests.test_case import TestCase


class Gemma4AudioConverterTest(TestCase):
    """Tests for Gemma4AudioConverter.

    Small parameters so tests run quickly:
        sampling_rate    = 100 Hz
        num_fft_bins     = 8
        stride           = 2
        max_audio_length = 1 second  → num_samples = 100
        num_mels         = 8
        max_frequency    = 50 Hz     (Nyquist at 100 Hz)
        num_frames       = num_samples // stride = 100 // 2 = 50
    """

    def setUp(self):
        self.init_kwargs = {
            "num_mels": 8,
            "num_fft_bins": 8,
            "stride": 2,
            "sampling_rate": 100,
            "max_audio_length": 1,
            "min_frequency": 0.0,
            "max_frequency": 50.0,
            "mel_floor": 1e-5,
        }
        # Fixed-length input matching num_samples = 100.
        self.num_samples = 100  # sampling_rate * max_audio_length
        self.num_frames = self.num_samples // 2  # stride = 2 → 50 frames
        self.input_data = np.ones((2, self.num_samples), dtype="float32")

    def test_audio_converter_basics(self):
        """Converter initialises and produces the correct output shape."""
        converter = Gemma4AudioConverter(**self.init_kwargs)
        out = converter(self.input_data)
        self.assertEqual(out.shape, (2, self.num_frames, 8))

    def test_1d_input_shape(self):
        """Single waveform (no batch dim) returns (num_frames, num_mels)."""
        converter = Gemma4AudioConverter(**self.init_kwargs)
        waveform = np.ones((self.num_samples,), dtype="float32")
        out = converter(waveform)
        self.assertEqual(out.shape, (self.num_frames, 8))

    def test_2d_input_shape(self):
        """Batched waveform returns (batch_size, num_frames, num_mels)."""
        converter = Gemma4AudioConverter(**self.init_kwargs)
        waveform = np.ones((3, self.num_samples), dtype="float32")
        out = converter(waveform)
        self.assertEqual(out.shape, (3, self.num_frames, 8))

    def test_short_input_is_padded(self):
        """Audio shorter than num_samples is zero-padded to the fixed length."""
        converter = Gemma4AudioConverter(**self.init_kwargs)
        short = np.ones((40,), dtype="float32")  # shorter than 100
        out = converter(short)
        self.assertEqual(out.shape, (self.num_frames, 8))

    def test_long_input_is_trimmed(self):
        """Audio longer than num_samples is trimmed to the fixed length."""
        converter = Gemma4AudioConverter(**self.init_kwargs)
        long = np.ones((200,), dtype="float32")  # longer than 100
        out = converter(long)
        self.assertEqual(out.shape, (self.num_frames, 8))

    def test_audio_shape_property(self):
        converter = Gemma4AudioConverter(**self.init_kwargs)
        self.assertEqual(converter.audio_shape(), (self.num_frames, 8))

    def test_output_is_finite(self):
        """Log-mel outputs should be finite and above log(mel_floor)."""
        import keras

        converter = Gemma4AudioConverter(**self.init_kwargs)
        waveform = (
            np.random.default_rng(42)
            .standard_normal(self.num_samples)
            .astype("float32")
        )
        out = converter(waveform)
        self.assertFalse(
            bool(keras.ops.any(keras.ops.isnan(out))), "Output contains NaN"
        )
        self.assertFalse(
            bool(keras.ops.any(keras.ops.isinf(out))), "Output contains Inf"
        )
        min_val = float(keras.ops.min(out))
        expected_min = np.log(self.init_kwargs["mel_floor"])
        self.assertGreaterEqual(min_val, expected_min - 1e-4)

    def test_zeros_output_equals_floor(self):
        """A silent (zero) input should produce log(mel_floor) everywhere."""
        converter = Gemma4AudioConverter(**self.init_kwargs)
        waveform = np.zeros((self.num_samples,), dtype="float32")
        out = ops.convert_to_numpy(converter(waveform))
        expected = float(np.log(self.init_kwargs["mel_floor"]))
        self.assertAllClose(out, expected * np.ones_like(out), atol=1e-4)

    def test_per_bin_mean_normalisation(self):
        """Per-bin mean subtraction should shift all outputs by -mean."""
        per_bin_mean = [1.0] * 8
        converter_base = Gemma4AudioConverter(**self.init_kwargs)
        converter_norm = Gemma4AudioConverter(
            **{**self.init_kwargs, "per_bin_mean": per_bin_mean}
        )
        waveform = np.ones((self.num_samples,), dtype="float32")
        out_base = ops.convert_to_numpy(converter_base(waveform))
        out_norm = ops.convert_to_numpy(converter_norm(waveform))
        self.assertAllClose(out_norm, out_base - 1.0, atol=1e-5)

    def test_per_bin_stddev_normalisation(self):
        """Per-bin stddev scaling should divide all outputs by stddev."""
        per_bin_stddev = [2.0] * 8
        converter_base = Gemma4AudioConverter(**self.init_kwargs)
        converter_norm = Gemma4AudioConverter(
            **{**self.init_kwargs, "per_bin_stddev": per_bin_stddev}
        )
        waveform = np.ones((self.num_samples,), dtype="float32")
        out_base = ops.convert_to_numpy(converter_base(waveform))
        out_norm = ops.convert_to_numpy(converter_norm(waveform))
        self.assertAllClose(out_norm, out_base / 2.0, atol=1e-5)

    def test_get_config_round_trip(self):
        """get_config / from_config should reproduce identical parameters."""
        converter = Gemma4AudioConverter(**self.init_kwargs)
        config = converter.get_config()
        restored = Gemma4AudioConverter.from_config(config)
        for key, val in self.init_kwargs.items():
            self.assertEqual(getattr(restored, key), val)

    def test_get_config_with_per_bin_normalisation(self):
        """Per-bin mean/stddev survive a config round-trip."""
        mean = list(range(8))
        stddev = [float(i + 1) for i in range(8)]
        converter = Gemma4AudioConverter(
            **{
                **self.init_kwargs,
                "per_bin_mean": mean,
                "per_bin_stddev": stddev,
            }
        )
        config = converter.get_config()
        self.assertEqual(config["per_bin_mean"], mean)
        self.assertEqual(config["per_bin_stddev"], stddev)

        restored = Gemma4AudioConverter.from_config(config)
        waveform = np.ones((self.num_samples,), dtype="float32")
        self.assertAllClose(
            ops.convert_to_numpy(converter(waveform)),
            ops.convert_to_numpy(restored(waveform)),
            atol=1e-5,
        )

    def test_default_parameters(self):
        """Verify the default parameter values."""
        converter = Gemma4AudioConverter()
        self.assertEqual(converter.num_mels, 128)
        self.assertEqual(converter.num_fft_bins, 400)
        self.assertEqual(converter.stride, 160)
        self.assertEqual(converter.sampling_rate, 16000)
        self.assertEqual(converter.max_audio_length, 30)
        self.assertAlmostEqual(converter.min_frequency, 0.0)
        self.assertAlmostEqual(converter.max_frequency, 8000.0)
        self.assertAlmostEqual(converter.mel_floor, 1e-5)
        self.assertIsNone(converter.per_bin_mean)
        self.assertIsNone(converter.per_bin_stddev)
        # num_samples = 16000 * 30 = 480_000
        self.assertEqual(converter.num_samples, 480_000)

    def test_default_output_shape(self):
        """1 second of 16 kHz audio padded to max_audio_length * sample_rate."""
        converter = Gemma4AudioConverter()
        waveform = np.zeros(16000, dtype=np.float32)
        out = converter(waveform)
        # num_frames = (16000 * 30) // 160 = 3000
        self.assertEqual(out.shape[-1], 128)
        self.assertEqual(out.shape[-2], 3000)
