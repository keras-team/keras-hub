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

"""Tests for Whisper preprocessing layers."""
import os

import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models.whisper.whisper_tokenizer import WhisperTokenizer
from keras_nlp.tests.test_case import TestCase


class WhisperTokenizerTest(TestCase):
    def setUp(self):
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

    def test_tokenize(self):
        input_data = " airplane at airport"
        output = self.tokenizer(input_data)
        self.assertAllEqual(output, [0, 1, 2, 0, 3])

    def test_tokenize_batch(self):
        input_data = tf.constant([" airplane at airport", " kohli is the best"])
        output = self.tokenizer(input_data)
        self.assertAllEqual(output, [[0, 1, 2, 0, 3], [4, 5, 6, 7, 8]])

    def test_detokenize(self):
        input_tokens = [0, 1, 2, 0, 3]
        output = self.tokenizer.detokenize(input_tokens)
        self.assertEqual(output, " airplane at airport")

    def test_detokenize_with_special_tokens(self):
        input_tokens = [9, 14, 12, 11, 0, 1, 2, 0, 3, 10]
        output = self.tokenizer.detokenize(input_tokens)
        print(output)
        self.assertEqual(
            output,
            "<|startoftranscript|><|en|><|transcribe|><|notimestamps|> airplane at airport<|endoftext|>",
        )

    def test_vocabulary_size(self):
        self.assertEqual(self.tokenizer.vocabulary_size(), 16)

    def test_special_tokens(self):
        self.assertEqual(self.tokenizer.bos_token_id, 9)
        self.assertEqual(self.tokenizer.eos_token_id, 10)
        self.assertEqual(self.tokenizer.pad_token_id, 10)
        self.assertEqual(self.tokenizer.no_timestamps_token_id, 11)
        self.assertEqual(self.tokenizer.translate_token_id, 13)
        self.assertEqual(self.tokenizer.transcribe_token_id, 12)

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            WhisperTokenizer(
                vocabulary=["a", "b", "c"], merges=[], special_tokens={}
            )

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saved_model(self, save_format, filename):
        input_data = tf.constant([" airplane at airport"])

        inputs = keras.Input(dtype="string", shape=())
        outputs = self.tokenizer(inputs)
        model = keras.Model(inputs, outputs)

        path = os.path.join(self.get_temp_dir(), filename)
        # Don't save traces in the tf format, we check compilation elsewhere.
        kwargs = {"save_traces": False} if save_format == "tf" else {}
        model.save(path, save_format=save_format, **kwargs)

        restored_model = keras.models.load_model(path)
        self.assertAllEqual(
            model(input_data),
            restored_model(input_data),
        )
