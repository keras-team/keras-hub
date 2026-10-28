import grain
import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_converter import (
    Qwen3ASRAudioConverter,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.keras_utils import running_on_gpu
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

    def test_feature_extractor_basics(self):
        if tf is None:
            self.skipTest("TensorFlow is not installed")

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

    def test_python_matches_tf(self):
        if tf is None:
            pytest.skip("TensorFlow is not installed")

        converter = Qwen3ASRAudioConverter(min_length=8000, n_window=50)
        rng = np.random.default_rng(42)
        audio = [
            rng.uniform(-1.0, 1.0, size=(4000,)).astype("float32"),
            rng.uniform(-1.0, 1.0, size=(12000,)).astype("float32"),
        ]

        python_output = converter._call_python(audio)
        tf_output = converter._call_tf(audio)
        atol = 1e-3 if running_on_gpu() else 1e-4
        self.assertAllClose(
            ops.convert_to_numpy(tf_output),
            python_output,
            atol=atol,
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
