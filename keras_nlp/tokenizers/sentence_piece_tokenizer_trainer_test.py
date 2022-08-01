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
"""Tests for WordPiece Tokenizer Trainer."""

import os

import tensorflow as tf

from keras_nlp.tokenizers.sentence_piece_tokenizer import SentencePieceTokenizer
from keras_nlp.tokenizers.sentence_piece_tokenizer_trainer import (
    compute_sentencepiece_vocabulary,
)


class SentencePieceTokenizerTrainerTest(tf.test.TestCase):
    def test_dataset_input(self):
        test_text = ["Ninjas and Samurais"]
        expected_output = [
            [5, 9, 6, 7, 11, 4, 8, 5, 4, 7, 10, 5, 12, 4, 13, 15, 14, 4, 6, 8]
        ]
        data = tf.data.Dataset.from_tensor_slices(test_text)
        proto = compute_sentencepiece_vocabulary(data, 16)
        tokenizer = SentencePieceTokenizer(proto=proto)
        test_output = tokenizer(test_text)
        self.assertAllEqual(expected_output, test_output)

    def test_filenames_input(self):
        test_text = "Ninja Land"
        with open("test.txt", "w+") as f:
            f.write(test_text + "\n")
        expected_output = [
            5,
            9,
            6,
            7,
            11,
            4,
            8,
            5,
            4,
            7,
            10,
            5,
            12,
            4,
            13,
            15,
            14,
            4,
            6,
            8,
        ]
        proto = compute_sentencepiece_vocabulary(
            ["test.txt"],
            16,
        )
        tokenizer = SentencePieceTokenizer(proto=proto)
        test_output = tokenizer(test_text)
        self.assertAllEqual(expected_output, test_output)
        os.remove("test.txt")

    def test_invalid_input(self):
        test_text_invalid = {"file1": "test.txt"}
        with self.assertRaisesRegex(
            ValueError,
            "The `data` argument must be either `tf.data.Dataset` or `list`. "
            f"Received: {type(test_text_invalid)}.",
        ):
            compute_sentencepiece_vocabulary(test_text_invalid, 10)

    def test_lowercase(self):
        inputs = tf.data.Dataset.from_tensor_slices(["Drifting Along"])
        proto = compute_sentencepiece_vocabulary(
            inputs, vocabulary_size=15, lowercase=True
        )
        tokenizer = SentencePieceTokenizer(proto=proto)
        output = inputs.map(tokenizer).take(1).get_single_element()
        expected_output = [4, 8, 12, 5, 9, 14, 5, 6, 13, 4, 7, 10, 11, 6, 13]
        self.assertAllEqual(expected_output, output)

    def test_proto_output_file(self):
        inputs = tf.data.Dataset.from_tensor_slices(["Drifting Along"])
        compute_sentencepiece_vocabulary(
            inputs, vocabulary_size=15, proto_output_file="model.spm"
        )
        tokenizer = SentencePieceTokenizer(proto="model.spm")
        output = inputs.map(tokenizer).take(1).get_single_element()
        expected_output = [4, 8, 12, 5, 9, 14, 5, 6, 13, 4, 7, 10, 11, 6, 13]
        self.assertAllEqual(expected_output, output)
        os.remove("model.spm")
