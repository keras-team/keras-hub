# Copyright 2023 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "To use `keras_nlp`, please install Tensorflow: `pip install tensorflow`. "
        "The TensorFlow package is required for data preprocessing with any backend."
    )

from keras_nlp.src.models.whisper.whisper_audio_feature_extractor import (
    WhisperAudioFeatureExtractor,
)
from keras_nlp.src.tests.test_case import TestCase


class WhisperAudioFeatureExtractorTest(TestCase):
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
            cls=WhisperAudioFeatureExtractor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_correctness(self):
        audio_tensor = tf.ones((2,), dtype="float32")
        outputs = WhisperAudioFeatureExtractor(**self.init_kwargs)(audio_tensor)

        # Verify shape.
        self.assertEqual(outputs.shape, (5, 80))
        # Verify output.
        expected = [1.1656, 1.0151, -0.8343, -0.8343, -0.8343]
        self.assertAllClose(outputs[:, 0], expected, atol=0.01, rtol=0.01)
