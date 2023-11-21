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

import os

import tensorflow as tf

from keras_nlp.tests.test_case import TestCase
from keras_nlp.tokenizers.sentence_piece_tokenizer import SentencePieceTokenizer


class SentencePieceTokenizerTest(TestCase):
    def setUp(self):
        super().setUp()
        self.proto = os.path.join(
            self.get_test_data_dir(), "tokenizer_test_vocab.spm"
        )

    def test_tokenize(self):
        input_data = ["the quick brown fox."]
        tokenizer = SentencePieceTokenizer(
            proto=self.proto,
        )
        call_output = tokenizer(input_data)
        tokenize_output = tokenizer.tokenize(input_data)
        self.assertAllEqual(call_output, [[6, 5, 3, 4]])
        self.assertAllEqual(tokenize_output, [[6, 5, 3, 4]])

    def test_scalar_tokenize(self):
        input_data = "the quick brown fox."
        tokenizer = SentencePieceTokenizer(
            proto=self.proto,
        )
        call_output = tokenizer(input_data)
        tokenize_output = tokenizer.tokenize(input_data)
        self.assertAllEqual(call_output, [6, 5, 3, 4])
        self.assertAllEqual(tokenize_output, [6, 5, 3, 4])

    def test_dense_output(self):
        input_data = ["the quick brown fox."]
        tokenizer = SentencePieceTokenizer(
            proto=self.proto,
            sequence_length=10,
        )
        output_data = tokenizer(input_data)
        self.assertAllEqual(output_data, [[6, 5, 3, 4, 0, 0, 0, 0, 0, 0]])

    def test_string_tokenize(self):
        input_data = ["the quick brown fox."]
        tokenizer = SentencePieceTokenizer(
            proto=self.proto,
            dtype="string",
        )
        output_data = tokenizer(input_data)
        self.assertAllEqual(
            output_data,
            [["▁the", "▁quick", "▁brown", "▁fox."]],
        )

    def test_detokenize(self):
        tokenizer = SentencePieceTokenizer(proto=self.proto)
        outputs = tokenizer.detokenize([6, 5, 3, 4])
        self.assertAllEqual(outputs, "the quick brown fox.")
        outputs = tokenizer.detokenize([[6, 5, 3, 4], [6, 4]])
        self.assertAllEqual(outputs, ["the quick brown fox.", "the fox."])

    def test_accessors(self):
        tokenizer = SentencePieceTokenizer(
            proto=self.proto,
        )
        self.assertEqual(
            tokenizer.get_vocabulary(),
            ["<unk>", "<s>", "</s>", "▁brown", "▁fox.", "▁quick", "▁the"],
        )
        self.assertEqual(type(tokenizer.get_vocabulary()), list)
        self.assertEqual(tokenizer.vocabulary_size(), 7)
        self.assertEqual(type(tokenizer.vocabulary_size()), int)
        self.assertEqual(tokenizer.id_to_token(0), "<unk>")
        self.assertEqual(tokenizer.id_to_token(5), "▁quick")
        self.assertEqual(type(tokenizer.id_to_token(0)), str)
        self.assertEqual(tokenizer.token_to_id("<unk>"), 0)
        self.assertEqual(tokenizer.token_to_id("▁quick"), 5)
        self.assertEqual(type(tokenizer.token_to_id("<unk>")), int)

    def test_error_id_out_of_vocabulary(self):
        tokenizer = SentencePieceTokenizer(
            proto=self.proto,
        )
        with self.assertRaises(ValueError):
            tokenizer.id_to_token(tokenizer.vocabulary_size())
        with self.assertRaises(ValueError):
            tokenizer.id_to_token(-1)

    def test_from_bytes(self):
        with tf.io.gfile.GFile(self.proto, "rb") as file:
            proto = file.read()
        tokenizer = SentencePieceTokenizer(
            proto=proto,
        )
        output_data = tokenizer(["the quick brown fox."])
        self.assertAllEqual(output_data, [[6, 5, 3, 4]])

    def test_tokenize_then_batch(self):
        tokenizer = SentencePieceTokenizer(
            proto=self.proto,
        )

        ds = tf.data.Dataset.from_tensor_slices(
            ["the quick brown fox.", "the quick", "the", "quick brown fox."]
        )
        ds = ds.map(tokenizer).apply(
            tf.data.experimental.dense_to_ragged_batch(4)
        )
        output_data = ds.take(1).get_single_element()

        expected = [
            [6, 5, 3, 4],
            [6, 5],
            [6],
            [5, 3, 4],
        ]
        for i in range(4):
            self.assertAllEqual(output_data[i], expected[i])

    def test_batch_then_tokenize(self):
        tokenizer = SentencePieceTokenizer(
            proto=self.proto,
        )

        ds = tf.data.Dataset.from_tensor_slices(
            ["the quick brown fox.", "the quick", "the", "quick brown fox."]
        )
        ds = ds.batch(4).map(tokenizer)
        output_data = ds.take(1).get_single_element()

        expected = [
            [6, 5, 3, 4],
            [6, 5],
            [6],
            [5, 3, 4],
        ]
        for i in range(4):
            self.assertAllEqual(output_data[i], expected[i])

    def test_config(self):
        input_data = ["the quick brown whale."]
        original_tokenizer = SentencePieceTokenizer(
            proto=self.proto,
        )
        cloned_tokenizer = SentencePieceTokenizer.from_config(
            original_tokenizer.get_config()
        )
        cloned_tokenizer.set_proto(original_tokenizer.proto)
        self.assertAllEqual(
            original_tokenizer(input_data),
            cloned_tokenizer(input_data),
        )
