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

"""Tests for Whisper preprocessor layer."""


import tensorflow as tf

from keras_nlp.backend import keras
from keras_nlp.models.whisper.whisper_audio_feature_extractor import (
    WhisperAudioFeatureExtractor,
)
from keras_nlp.models.whisper.whisper_preprocessor import WhisperPreprocessor
from keras_nlp.models.whisper.whisper_tokenizer import WhisperTokenizer
from keras_nlp.tests.test_case import TestCase


class WhisperPreprocessorTest(TestCase):
    def setUp(self):
        self.num_mels = 80
        self.num_fft_bins = 400
        self.stride = 100
        self.sampling_rate = 100
        self.max_audio_length = 5
        self.output_length = (
            self.max_audio_length * self.sampling_rate
        ) // self.stride
        self.audio_feature_extractor = WhisperAudioFeatureExtractor(
            num_mels=self.num_mels,
            num_fft_bins=self.num_fft_bins,
            stride=self.stride,
            sampling_rate=self.sampling_rate,
            max_audio_length=self.max_audio_length,
        )

        self.vocab = {
            "Ġair": 0,
            "plane": 1,
            "Ġat": 2,
            "port": 3,
            "Ġkoh": 4,
            "li": 5,
            "Ġis": 6,
            "Ġthe": 7,
            "Ġbest": 8,
        }

        merges = ["Ġ a", "Ġ t", "Ġ k", "Ġ i", "Ġ b", "Ġa i", "p l", "n e"]
        merges += ["Ġa t", "p o", "r t", "o h", "l i", "Ġi s", "Ġb e", "s t"]
        merges += ["Ġt h", "Ġai r", "pl a", "Ġk oh", "Ġth e", "Ġbe st", "po rt"]
        merges += ["pla ne"]
        self.merges = merges

        self.special_tokens = {
            "<|startoftranscript|>": 9,
            "<|endoftext|>": 10,
            "<|notimestamps|>": 11,
            "<|transcribe|>": 12,
            "<|translate|>": 13,
        }

        self.language_tokens = {
            "<|en|>": 14,
            "<|fr|>": 15,
        }

        self.tokenizer = WhisperTokenizer(
            vocabulary=self.vocab,
            merges=self.merges,
            special_tokens=self.special_tokens,
            language_tokens=self.language_tokens,
        )

        self.preprocessor = WhisperPreprocessor(
            audio_feature_extractor=self.audio_feature_extractor,
            tokenizer=self.tokenizer,
            decoder_sequence_length=12,
            language="<|en|>",
            task="translate",
        )

    def test_unbatched_preprocess(self):
        input_data = {
            "encoder_audio": tf.ones((200,)),
            "decoder_text": tf.constant(" airplane at airport"),
        }

        x = self.preprocessor(input_data)
        self.assertAllEqual(
            x["encoder_features"].shape, [self.output_length, self.num_mels]
        )
        self.assertAllEqual(
            x["decoder_token_ids"], [9, 14, 13, 11, 0, 1, 2, 0, 3, 10, 10, 10]
        )
        self.assertAllEqual(
            x["decoder_padding_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
        )

    def test_preprocess_batch(self):
        input_data = {
            "encoder_audio": tf.ones((4, 200)),
            "decoder_text": tf.constant([" airplane at airport"] * 4),
        }

        x = self.preprocessor(input_data)
        self.assertAllEqual(
            x["encoder_features"].shape, [4, self.output_length, self.num_mels]
        )
        self.assertAllEqual(
            x["decoder_token_ids"],
            [[9, 14, 13, 11, 0, 1, 2, 0, 3, 10, 10, 10]] * 4,
        )
        self.assertAllEqual(
            x["decoder_padding_mask"],
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]] * 4,
        )

    def test_preprocess_labeled_batch(self):
        x = {
            "encoder_audio": tf.ones((4, 200)),
            "decoder_text": tf.constant([" airplane at airport"] * 4),
        }
        y_in = tf.constant([1] * 4)
        sw_in = tf.constant([1.0] * 4)
        x, y, sw = self.preprocessor(x, y_in, sw_in)
        self.assertAllEqual(
            x["encoder_features"].shape, [4, self.output_length, self.num_mels]
        )
        self.assertAllEqual(
            x["decoder_token_ids"],
            [[9, 14, 13, 11, 0, 1, 2, 0, 3, 10, 10, 10]] * 4,
        )
        self.assertAllEqual(
            x["decoder_padding_mask"],
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]] * 4,
        )
        self.assertAllEqual(y, y_in)
        self.assertAllEqual(sw, sw_in)

    def test_preprocess_dataset(self):
        x = {
            "encoder_audio": tf.ones((4, 200)),
            "decoder_text": tf.constant([" airplane at airport"] * 4),
        }
        ds = tf.data.Dataset.from_tensor_slices(x)
        ds = ds.map(self.preprocessor)
        x = ds.batch(4).take(1).get_single_element()
        self.assertAllEqual(
            x["encoder_features"].shape, [4, self.output_length, self.num_mels]
        )
        self.assertAllEqual(
            x["decoder_token_ids"],
            [[9, 14, 13, 11, 0, 1, 2, 0, 3, 10, 10, 10]] * 4,
        )
        self.assertAllEqual(
            x["decoder_padding_mask"],
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]] * 4,
        )

    def test_sequence_length_override(self):
        input_data = {
            "encoder_audio": tf.ones((200,)),
            "decoder_text": tf.constant(" airplane at airport"),
        }
        x = self.preprocessor(input_data, decoder_sequence_length=6)
        self.assertAllEqual(x["decoder_token_ids"], [9, 14, 13, 11, 0, 10])

    def test_serialization(self):
        config = keras.saving.serialize_keras_object(self.preprocessor)
        new_preprocessor = keras.saving.deserialize_keras_object(config)
        self.assertEqual(
            new_preprocessor.get_config(),
            self.preprocessor.get_config(),
        )
