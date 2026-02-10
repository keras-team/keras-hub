import numpy as np

from keras_hub.src.models.gemma3n.gemma3n_audio_converter import (
    Gemma3nAudioConverter,
)
from keras_hub.src.tests.test_case import TestCase


class Gemma3nAudioConverterTest(TestCase):
    def setUp(self):
        super().setUp()
        self.feature_size = 128
        self.sampling_rate = 16000
        self.hop_length_ms = 10.0
        self.frame_length_ms = 32.0
        # Dummy audio.
        self.input_data = [
            np.sin(
                2
                * np.pi
                * 440
                * np.linspace(0, 1, self.sampling_rate, dtype=np.float32)
            )
        ]
        self.init_kwargs = {
            "feature_size": self.feature_size,
            "sampling_rate": self.sampling_rate,
            "padding_value": 0.0,
            "return_attention_mask": True,
            "frame_length_ms": self.frame_length_ms,
            "hop_length_ms": self.hop_length_ms,
            "min_frequency": 125.0,
            "max_frequency": 7600.0,
            "preemphasis": 0.97,
            "preemphasis_htk_flavor": True,
            "fft_overdrive": True,
            "dither": 0.0,
            "input_scale_factor": 1.0,
            "mel_floor": 1e-5,
            "per_bin_mean": None,
            "per_bin_stddev": None,
            "padding_side": "right",
        }

    def test_output_shape(self):
        converter = Gemma3nAudioConverter(**self.init_kwargs)
        outputs = converter(self.input_data[0])
        frame_length = int(
            round(self.sampling_rate * self.frame_length_ms / 1000.0)
        )
        hop_length = int(
            round(self.sampling_rate * self.hop_length_ms / 1000.0)
        )
        num_frames = (len(self.input_data[0]) - frame_length) // hop_length + 1
        expected_features_shape = (num_frames, self.feature_size)
        expected_mask_shape = (num_frames,)
        # Check that the outputs are tuples with two elements.
        self.assertIsInstance(outputs, tuple)
        self.assertEqual(len(outputs), 2)
        input_features, input_features_mask = outputs
        self.assertEqual(input_features.shape, expected_features_shape)
        self.assertEqual(input_features_mask.shape, expected_mask_shape)

    def test_padding(self):
        max_length = 20000
        pad_to_multiple_of = 128
        converter = Gemma3nAudioConverter(**self.init_kwargs)
        outputs = converter(
            self.input_data[0],
            padding="max_length",
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        # Calculate expectations.
        if max_length % pad_to_multiple_of != 0:
            padded_length = (
                (max_length // pad_to_multiple_of) + 1
            ) * pad_to_multiple_of
        else:
            padded_length = max_length
        frame_length = int(
            round(self.sampling_rate * self.frame_length_ms / 1000.0)
        )
        hop_length = int(
            round(self.sampling_rate * self.hop_length_ms / 1000.0)
        )
        num_frames = (padded_length - frame_length) // hop_length + 1
        expected_features_shape = (num_frames, self.feature_size)
        # Check that the outputs are tuples with two elements.
        self.assertIsInstance(outputs, tuple)
        self.assertEqual(len(outputs), 2)
        input_features, _ = outputs
        self.assertEqual(input_features.shape, expected_features_shape)

    def test_normalization(self):
        mean = np.random.rand(self.feature_size).tolist()
        stddev = np.random.rand(self.feature_size).tolist()
        # One converter with normalization and one without.
        converter_no_norm = Gemma3nAudioConverter(**self.init_kwargs)
        norm_kwargs = self.init_kwargs.copy()
        norm_kwargs["per_bin_mean"] = mean
        norm_kwargs["per_bin_stddev"] = stddev
        converter_norm = Gemma3nAudioConverter(**norm_kwargs)
        outputs_no_norm = converter_no_norm(self.input_data)
        outputs_norm = converter_norm(self.input_data)
        # Check that the outputs are tuples with two elements.
        self.assertIsInstance(outputs_no_norm, tuple)
        self.assertEqual(len(outputs_no_norm), 2)
        self.assertIsInstance(outputs_norm, tuple)
        self.assertEqual(len(outputs_norm), 2)
        features_no_norm, _ = outputs_no_norm
        features_norm, _ = outputs_norm
        # We would want outputs to be different.
        self.assertNotAllClose(features_no_norm, features_norm)
        # Manually normalize and check for closeness.
        manual_norm_features = (features_no_norm - np.array(mean)) / np.array(
            stddev
        )
        self.assertAllClose(manual_norm_features, features_norm)

    def test_serialization(self):
        instance = Gemma3nAudioConverter(**self.init_kwargs)
        self.run_serialization_test(instance=instance)
