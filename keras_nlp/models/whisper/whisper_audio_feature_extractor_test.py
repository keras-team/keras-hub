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
"""Tests for Whisper audio feature extractor."""

import tensorflow as tf

from keras_nlp.models.whisper.whisper_audio_feature_extractor import NUM_MELS
from keras_nlp.models.whisper.whisper_audio_feature_extractor import (
    WhisperAudioFeatureExtractor,
)


class WhisperAudioFeatureExtractorTest(tf.test.TestCase):
    def setUp(self):
        self.sample_rate = 100
        self.num_fft_bins = 400
        self.stride = 100
        self.max_audio_length = 5
        self.whisper_audio_feature_extractor = WhisperAudioFeatureExtractor(
            sample_rate=self.sample_rate,
            num_fft_bins=self.num_fft_bins,
            stride=self.stride,
            max_audio_length=self.max_audio_length,
        )

    def test_unbatched_inputs(self):
        audio_tensor = tf.ones((2,), dtype="float32")

        outputs = self.whisper_audio_feature_extractor(audio_tensor)

        # Verify shape.
        self.assertEqual(outputs.shape, (1, 5, NUM_MELS))

        # Verify output.
        expected = [1.1656, 1.0151, -0.8343, -0.8343, -0.8343]
        self.assertAllClose(outputs[0, :, 0], expected, atol=0.01, rtol=0.01)

    def test_batched_inputs(self):
        audio_tensor_1 = tf.ones((2,), dtype="float32")
        audio_tensor_2 = tf.ones((25,), dtype="float32")
        audio_tensor = tf.ragged.stack([audio_tensor_1, audio_tensor_2], axis=0)

        outputs = self.whisper_audio_feature_extractor(audio_tensor)

        # Verify shape.
        self.assertEqual(outputs.shape, (2, 5, NUM_MELS))
