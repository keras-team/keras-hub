# Copyright 2022 The KerasNLP Authors
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

"""Tests for XLM-RoBERTa tokenizer."""

import io
import os

import sentencepiece
import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models.xlm_roberta.xlm_roberta_preprocessor import (
    XLMRobertaTokenizer,
)


class XLMRobertaTokenizerTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        bytes_io = io.BytesIO()
        vocab_data = tf.data.Dataset.from_tensor_slices(
            ["the quick brown fox", "the earth is round"]
        )
        sentencepiece.SentencePieceTrainer.train(
            sentence_iterator=vocab_data.as_numpy_iterator(),
            model_writer=bytes_io,
            vocab_size=10,
            model_type="WORD",
            unk_id=0,
            bos_id=1,
            eos_id=2,
        )
        self.proto = bytes_io.getvalue()

        self.tokenizer = XLMRobertaTokenizer(proto=self.proto)

    def test_tokenize(self):
        input_data = "the quick brown fox"
        output = self.tokenizer(input_data)
        self.assertAllEqual(output, [4, 9, 5, 7])

    def test_tokenize_batch(self):
        input_data = tf.constant(["the quick brown fox", "the earth is round"])
        output = self.tokenizer(input_data)
        self.assertAllEqual(output, [[4, 9, 5, 7], [4, 6, 8, 10]])

    def test_unk_token(self):
        input_data = "the quick brown fox running"

        output = self.tokenizer(input_data)
        self.assertAllEqual(output, [4, 9, 5, 7, 3])

    def test_detokenize(self):
        input_data = tf.constant([[4, 9, 5, 7]])
        output = self.tokenizer.detokenize(input_data)
        self.assertEqual(output, tf.constant(["the quick brown fox"]))

    def test_vocabulary(self):
        vocabulary = self.tokenizer.get_vocabulary()
        self.assertAllEqual(
            vocabulary,
            [
                "<s>",
                "<pad>",
                "</s>",
                "<unk>",
                "▁the",
                "▁brown",
                "▁earth",
                "▁fox",
                "▁is",
                "▁quick",
                "▁round",
            ],
        )
        self.assertEqual(self.tokenizer.vocabulary_size(), 11)

    def test_id_to_token(self):
        print(self.tokenizer.id_to_token(9))
        self.assertEqual(self.tokenizer.id_to_token(9), "▁quick")
        self.assertEqual(self.tokenizer.id_to_token(5), "▁brown")

    def test_token_to_id(self):
        self.assertEqual(self.tokenizer.token_to_id("▁the"), 4)
        self.assertEqual(self.tokenizer.token_to_id("▁round"), 10)

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saved_model(self, save_format, filename):
        input_data = tf.constant(["the quick brown fox"])

        inputs = keras.Input(dtype="string", shape=())
        outputs = self.tokenizer(inputs)
        model = keras.Model(inputs, outputs)

        path = os.path.join(self.get_temp_dir(), filename)
        model.save(path, save_format=save_format)

        restored_model = keras.models.load_model(path)
        self.assertAllEqual(
            model(input_data),
            restored_model(input_data),
        )
