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

import os

import pytest
import tensorflow as tf

from keras_nlp.backend import keras
from keras_nlp.models.whisper.whisper_audio_feature_extractor import (
    WhisperAudioFeatureExtractor,
)
from keras_nlp.tests.test_case import TestCase


class WhisperAudioFeatureExtractorTest(TestCase):
    def setUp(self):
        self.num_mels = 80
        self.num_fft_bins = 400
        self.stride = 100
        self.sampling_rate = 100
        self.max_audio_length = 5
        self.audio_feature_extractor = WhisperAudioFeatureExtractor(
            num_mels=self.num_mels,
            num_fft_bins=self.num_fft_bins,
            stride=self.stride,
            sampling_rate=self.sampling_rate,
            max_audio_length=self.max_audio_length,
        )

    def test_unbatched_inputs(self):
        audio_tensor = tf.ones((2,), dtype="float32")

        outputs = self.audio_feature_extractor(audio_tensor)

        # Verify shape.
        self.assertEqual(outputs.shape, (1, 5, self.num_mels))
        # Verify output.
        expected = [1.1656, 1.0151, -0.8343, -0.8343, -0.8343]
        self.assertAllClose(outputs[0, :, 0], expected, atol=0.01, rtol=0.01)

    def test_batched_inputs(self):
        audio_tensor_1 = tf.ones((2,), dtype="float32")
        audio_tensor_2 = tf.ones((25,), dtype="float32")
        audio_tensor = tf.ragged.stack([audio_tensor_1, audio_tensor_2], axis=0)

        outputs = self.audio_feature_extractor(audio_tensor)

        # Verify shape.
        self.assertEqual(outputs.shape, (2, 5, self.num_mels))
        # Verify output.
        expected_1 = [1.1656, 1.0151, -0.8343, -0.8343, -0.8343]
        self.assertAllClose(outputs[0, :, 0], expected_1, atol=0.01, rtol=0.01)
        expected_2 = [1.2299, 1.0970, 0.3997, -0.7700, -0.7700]
        self.assertAllClose(outputs[1, :, 0], expected_2, atol=0.01, rtol=0.01)

    def test_serialization(self):
        config = keras.saving.serialize_keras_object(
            self.audio_feature_extractor
        )
        new_audio_feature_extractor = keras.saving.deserialize_keras_object(
            config
        )
        self.assertEqual(
            new_audio_feature_extractor.get_config(),
            self.audio_feature_extractor.get_config(),
        )

    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_model(self):
        audio_tensor = tf.ones((2, 200), dtype="float32")

        inputs = keras.Input(dtype="float32", shape=(None,))
        outputs = self.audio_feature_extractor(inputs)
        model = keras.Model(inputs, outputs)

        path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(path, save_format="keras_v3")

        restored_model = keras.models.load_model(path)
        self.assertAllEqual(
            model(audio_tensor),
            restored_model(audio_tensor),
        )
