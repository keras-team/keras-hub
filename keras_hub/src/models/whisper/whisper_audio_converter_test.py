import grain
import numpy as np
import tensorflow as tf

from keras_hub.src.models.whisper.whisper_audio_converter import (
    WhisperAudioConverter,
)
from keras_hub.src.tests.test_case import TestCase


class WhisperAudioConverterTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "num_mels": 80,
            "num_fft_bins": 400,
            "stride": 100,
            "sampling_rate": 100,
            "max_audio_length": 5,
        }
        audio_tensor_1 = tf.ones((2,), dtype="float32")
        audio_tensor_2 = tf.ones((25,), dtype="float32")
        self.input_data = tf.ragged.stack(
            [audio_tensor_1, audio_tensor_2],
            axis=0,
        )

    def test_feature_extractor_basics(self):
        self.run_preprocessing_layer_test(
            cls=WhisperAudioConverter,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_correctness(self):
        audio_tensor = tf.ones((2,), dtype="float32")
        outputs = WhisperAudioConverter(**self.init_kwargs)(audio_tensor)

        # Verify shape.
        self.assertEqual(outputs.shape, (5, 80))
        # Verify output.
        expected = [1.1656, 1.0151, -0.8343, -0.8343, -0.8343]
        self.assertAllClose(outputs[:, 0], expected, atol=0.01, rtol=0.01)

    def test_python_matches_tf(self):
        converter = WhisperAudioConverter(**self.init_kwargs)
        audio = np.random.default_rng(42).random((2, 300)).astype("float32")
        self.assertAllClose(
            converter._call_python(audio),
            converter._call_tf(audio),
            atol=1e-4,
        )

    def test_grain_outputs_numpy(self):
        converter = WhisperAudioConverter(**self.init_kwargs)
        rng = np.random.default_rng(42)
        samples = [
            rng.random((300,)).astype("float32"),
            rng.random((120,)).astype("float32"),
        ]
        # Unbatched source. Grain returns numpy arrays, not backend tensors.
        ds = grain.MapDataset.source(samples).map(converter)
        for sample, output in zip(samples, ds):
            self.assertIsInstance(output, np.ndarray)
            self.assertEqual(output.shape, (5, 80))
            self.assertAllClose(output, converter(sample))
        # Batching after the map gives the same result as a batched call.
        (batch,) = list(ds.batch(2))
        self.assertIsInstance(batch, np.ndarray)
        self.assertEqual(batch.shape, (2, 5, 80))
        # Batched source.
        batched = np.stack(
            [np.pad(sample, (0, 300 - sample.shape[0])) for sample in samples]
        )
        (output,) = list(grain.MapDataset.source([batched]).map(converter))
        self.assertAllClose(output, batch)
