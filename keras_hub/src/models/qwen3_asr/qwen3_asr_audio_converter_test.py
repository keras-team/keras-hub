import grain
import keras
import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_converter import (
    Qwen3ASRAudioConverter,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.tensor_utils import tf


class Qwen3ASRAudioConverterTest(TestCase):
    def test_serialization(self):
        self.run_layer_test(
            cls=Qwen3ASRAudioConverter,
            init_kwargs={
                "num_mels": 4,
                "min_length": 8000,
            },
            input_data=np.ones((1, 8000), dtype="float32"),
            expected_output_shape=(1, 50, 4),
            run_training_check=False,
            run_precision_checks=False,
        )
        converter = Qwen3ASRAudioConverter(num_mels=4, min_length=8000)
        self.assertNotIn("max_audio_length", converter.get_config())
        with self.assertRaises(TypeError):
            Qwen3ASRAudioConverter(max_audio_length=1)

    def test_converter_call(self):
        # 1 second of audio at 16000Hz
        audio = np.random.uniform(size=(16000,))

        converter = Qwen3ASRAudioConverter(
            num_mels=128,
            sampling_rate=16000,
            n_window=50,
        )

        # Call with unbatched input
        output = converter(audio)
        self.assertEqual(output.shape, (100, 128))

        # Call with batched input
        batched_audio = np.random.uniform(size=(2, 16000))
        output_batched = converter(batched_audio)
        self.assertEqual(output_batched.shape, (2, 100, 128))

    def test_audio_shape(self):
        converter = Qwen3ASRAudioConverter(n_window=50)
        self.assertEqual(converter.audio_shape(), (None, 128))

    def test_variable_length_batched_input(self):
        converter = Qwen3ASRAudioConverter(n_window=50)
        audio = [np.ones((16000,)), np.ones((8000,))]

        output = converter(audio)

        self.assertEqual(output.shape, (2, 100, 128))

    def test_per_clip_mel_padding(self):
        converter = Qwen3ASRAudioConverter(min_length=8000, n_window=50)
        rng = np.random.default_rng(7)
        short_clip = rng.uniform(-1.0, 1.0, size=(9600,)).astype("float32")
        long_clip = rng.uniform(-1.0, 1.0, size=(16000,)).astype("float32")

        solo_short = converter._call_python(short_clip)
        batched = converter._call_python([short_clip, long_clip])
        self.assertAllClose(batched[0, :60], solo_short, atol=1e-6, rtol=1e-6)
        self.assertAllClose(
            batched[0, 60:], np.zeros((40, 128), dtype="float32"), atol=0.0
        )

        if tf is not None:
            tf_solo = ops.convert_to_numpy(converter._call_tf(short_clip))
            tf_batched = ops.convert_to_numpy(
                converter._call_tf([short_clip, long_clip])
            )
            self.assertAllClose(
                tf_batched[0, :60], tf_solo, atol=1e-6, rtol=1e-6
            )
            self.assertAllClose(
                tf_batched[0, 60:],
                np.zeros((40, 128), dtype="float32"),
                atol=0.0,
            )

    def test_min_length_padding(self):
        converter = Qwen3ASRAudioConverter(min_length=8000)
        audio = np.ones((4000,), dtype="float32")
        padded_audio = np.pad(audio, (0, 4000))

        output = converter._call_python(audio)
        expected_output = converter._call_python(padded_audio)

        self.assertEqual(output.shape, (50, 128))
        self.assertAllClose(output, expected_output, atol=0.0, rtol=0.0)

        if tf is not None:
            tf_output = ops.convert_to_numpy(converter._call_tf(audio))
            expected_tf_output = ops.convert_to_numpy(
                converter._call_tf(padded_audio)
            )
            self.assertAllClose(
                tf_output, expected_tf_output, atol=0.0, rtol=0.0
            )

    def test_python_path(self):
        converter = Qwen3ASRAudioConverter(min_length=8000)
        audio = [np.ones((4000,)), np.ones((8000,))]

        output = converter._call_python(audio)

        self.assertEqual(output.shape, (2, 50, 128))

    def test_rank_gt_2_and_odd_fft_bins(self):
        converter = Qwen3ASRAudioConverter(min_length=8000)
        with self.assertRaises(ValueError):
            converter(np.zeros((1, 2, 8000), dtype="float32"))
        with self.assertRaises(ValueError):
            converter.compute_output_spec(keras.KerasTensor((1, 2, 8000)))

        odd_converter = Qwen3ASRAudioConverter(
            num_fft_bins=401, min_length=8000
        )
        spec = odd_converter.compute_output_spec(
            keras.KerasTensor((None, 8000))
        )
        actual = odd_converter(np.zeros((1, 8000), dtype="float32"))
        self.assertEqual(spec.shape, (None, actual.shape[1], 128))

    @pytest.mark.skipif(tf is None, reason="TensorFlow is not installed")
    def test_feature_extractor_basics(self):
        audio = tf.constant(
            np.stack(
                [
                    np.pad(np.zeros((4000,), dtype="float32"), (0, 4000)),
                    np.zeros((8000,), dtype="float32"),
                ]
            )
        )
        with tf.device("/CPU:0"):
            self.run_preprocessing_layer_test(
                cls=Qwen3ASRAudioConverter,
                init_kwargs={"min_length": 8000},
                input_data=audio,
            )

    def test_audio_longer_than_default_max_length(self):
        converter = Qwen3ASRAudioConverter(n_window=50)
        audio = [np.zeros((5 * 16000,)), np.zeros((31 * 16000,))]

        output = converter(audio)

        self.assertEqual(output.shape, (2, 3100, 128))

    @pytest.mark.skipif(tf is None, reason="TensorFlow is not installed")
    def test_python_matches_tf(self):
        converter = Qwen3ASRAudioConverter(min_length=8000, n_window=50)
        rng = np.random.default_rng(42)
        audio = [
            rng.uniform(-1.0, 1.0, size=(4000,)).astype("float32"),
            rng.uniform(-1.0, 1.0, size=(12000,)).astype("float32"),
        ]

        python_output = converter._call_python(audio)
        tf_output = converter._call_tf(audio)
        self.assertAllClose(
            ops.convert_to_numpy(tf_output),
            python_output,
            atol=1e-4,
        )

    def test_grain_outputs_numpy(self):
        converter = Qwen3ASRAudioConverter(min_length=8000)
        rng = np.random.default_rng(42)
        samples = [
            rng.uniform(-1.0, 1.0, size=(4000,)).astype("float32"),
            rng.uniform(-1.0, 1.0, size=(6000,)).astype("float32"),
        ]

        dataset = grain.MapDataset.source(samples).map(converter)
        for sample, output in zip(samples, dataset):
            self.assertIsInstance(output, np.ndarray)
            self.assertEqual(output.shape, (50, 128))
            self.assertAllClose(output, converter(sample))

        (batch,) = list(dataset.batch(2))
        self.assertIsInstance(batch, np.ndarray)
        self.assertEqual(batch.shape, (2, 50, 128))

        batched_audio = np.stack(
            [np.pad(sample, (0, 6000 - len(sample))) for sample in samples]
        )
        (batched_output,) = list(
            grain.MapDataset.source([batched_audio]).map(converter)
        )
        self.assertAllClose(batched_output, batch)

        mixed_samples = [
            rng.uniform(-1.0, 1.0, size=(9600,)).astype("float32"),
            rng.uniform(-1.0, 1.0, size=(9600,)).astype("float32"),
        ]
        (mixed_batch,) = list(
            grain.MapDataset.source(mixed_samples).map(converter).batch(2)
        )
        self.assertAllClose(mixed_batch, converter(mixed_samples))
