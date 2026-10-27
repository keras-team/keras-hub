import grain
import keras
import numpy as np
from absl.testing import parameterized

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

    @parameterized.named_parameters(
        ("python", True),
        ("tf", False),
    )
    def test_dither_determinism(self, allow_python_workflow):
        kwargs = dict(self.init_kwargs)
        kwargs["dither"] = 1.0
        kwargs["seed"] = 42
        kwargs["_allow_python_workflow"] = allow_python_workflow
        converter = Gemma3nAudioConverter(**kwargs)

        # 1. Determinism
        audio_A = np.zeros(2000, dtype=np.float32)
        out1, _ = converter(audio_A)
        out2, _ = converter(audio_A)
        self.assertAllClose(out1, out2)

        # 2. Hash is load-bearing
        audio_B = np.zeros(2000, dtype=np.float32)
        audio_B[-1] = 1.0
        out3, _ = converter(audio_B)
        self.assertNotAllClose(out1[:2, :], out3[:2, :])

        # 3. Batch invariance: a record's features must not depend on the
        # other records it is batched with. `call()` cannot take a ragged
        # batch, so drive `pad()` + `_extract_spectrogram` directly, the
        # same way `_process_python` does.
        audio_C = np.zeros(4000, dtype=np.float32)
        padded_lengths = []
        batched_features = []
        for batch in ([audio_A], [audio_A, audio_C]):
            padded, masks = converter.pad(
                [record.reshape(-1, 1) for record in batch],
                padding="longest",
                max_length=480000,
                truncation=True,
                pad_to_multiple_of=128,
                return_attention_mask=True,
            )
            padded_lengths.append(len(padded[0]))
            features, _ = converter._extract_spectrogram(
                np.asarray(padded[0].T, dtype=converter.compute_dtype),
                np.asarray(masks[0], dtype="int32"),
            )
            batched_features.append(features)
        # `audio_A` pads to 2048 on its own, but to 4096 when batched next
        # to the 4000-sample `audio_C`.
        self.assertEqual(padded_lengths, [2048, 4096])
        self.assertAllClose(
            batched_features[0],
            batched_features[1][: batched_features[0].shape[0]],
        )

    def test_dither_negative_seed(self):
        # `record_hash` for the all-zeros 2000-sample record is 3125184870,
        # so a base seed of -3125184871 makes `self.seed + record_hash`
        # negative. `np.random.default_rng` rejects negative seeds, so the
        # derived seed must be wrapped into the unsigned 32-bit range.
        kwargs = dict(self.init_kwargs)
        kwargs["dither"] = 1.0
        kwargs["seed"] = -3125184871
        converter = Gemma3nAudioConverter(**kwargs)
        audio = np.zeros(2000, dtype=np.float32)
        out1, _ = converter(audio)
        out2, _ = converter(audio)
        self.assertEqual(out1.shape[-1], self.feature_size)
        self.assertAllClose(out1, out2)

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
        # Manually normalize and check for closeness. The layer returns
        # backend tensors, which on an accelerator cannot be mixed with NumPy
        # arrays directly, so bring them to NumPy first.
        features_no_norm = keras.ops.convert_to_numpy(features_no_norm)
        manual_norm_features = (features_no_norm - np.array(mean)) / np.array(
            stddev
        )
        self.assertAllClose(manual_norm_features, features_norm)

    def test_serialization(self):
        instance = Gemma3nAudioConverter(**self.init_kwargs)
        self.run_serialization_test(instance=instance)

    def test_python_matches_tf(self):
        converter = Gemma3nAudioConverter(**self.init_kwargs)
        audio = self.input_data[0][:4000]
        py_features, py_mask = converter._call_python(audio)
        tf_features, tf_mask = converter._call_tf(audio)
        self.assertAllClose(py_features, tf_features)
        self.assertAllEqual(py_mask, tf_mask)

    def test_grain_outputs_numpy(self):
        converter = Gemma3nAudioConverter(**self.init_kwargs)
        samples = [self.input_data[0][:4000], self.input_data[0][:8000]]
        ds = grain.MapDataset.source(samples).map(converter)
        for sample, output in zip(samples, ds):
            self.assertIsInstance(output, tuple)
            features, mask = output
            self.assertIsInstance(features, np.ndarray)
            self.assertIsInstance(mask, np.ndarray)
            self.assertEqual(features.shape[-1], self.feature_size)
            self.assertEqual(features.shape[0], mask.shape[0])
            expected_features, expected_mask = converter(sample)
            self.assertAllClose(features, expected_features)
            self.assertAllEqual(mask, expected_mask)
