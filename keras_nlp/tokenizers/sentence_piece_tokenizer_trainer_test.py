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
"""Tests for SentencePiece Tokenizer Trainer."""

import os
import re

import tensorflow as tf

from keras_nlp.tokenizers.sentence_piece_tokenizer import SentencePieceTokenizer
from keras_nlp.tokenizers.sentence_piece_tokenizer_trainer import (
    compute_sentence_piece_proto,
)


class SentencePieceTokenizerTrainerTest(tf.test.TestCase):
    def test_dataset_input(self):
        test_text = ["Ninjas and Samurais"]
        expected_output = [
            [5, 9, 6, 7, 11, 4, 8, 5, 4, 7, 10, 5, 12, 4, 13, 15, 14, 4, 6, 8]
        ]
        data = tf.data.Dataset.from_tensor_slices(test_text)
        proto = compute_sentence_piece_proto(data, 16)
        tokenizer = SentencePieceTokenizer(proto=proto)
        test_output = tokenizer(test_text)
        self.assertAllEqual(expected_output, test_output)

    def test_file_input(self):
        test_text = "Ninja Land"
        with open(os.path.join(self.get_temp_dir(), "test.txt"), "w+") as f:
            f.write(test_text + "\n")
        expected_output = [6, 8, 9, 5, 11, 4, 6, 7, 4, 5, 10]
        proto = compute_sentence_piece_proto(
            [os.path.join(self.get_temp_dir(), "test.txt")],
            12,
        )
        tokenizer = SentencePieceTokenizer(proto=proto)
        test_output = tokenizer(test_text)
        self.assertAllEqual(expected_output, test_output)

    def test_multiple_file_input(self):
        with open(os.path.join(self.get_temp_dir(), "test1.txt"), "w+") as f:
            f.write("Drifting Along\n")
        with open(os.path.join(self.get_temp_dir(), "test2.txt"), "w+") as f:
            f.write("Woah look there\n")
        inputs = [
            os.path.join(self.get_temp_dir(), "test1.txt"),
            os.path.join(self.get_temp_dir(), "test2.txt"),
        ]
        proto = compute_sentence_piece_proto(inputs, 20)
        tokenizer = SentencePieceTokenizer(proto=proto)
        test_text = "Woah Along"
        test_output = tokenizer(test_text)
        expected_output = [4, 16, 5, 17, 9, 4, 15, 12, 5, 11, 18]
        self.assertAllEqual(expected_output, test_output)

    def test_invalid_input(self):
        test_text_invalid = {"file1": "test.txt"}
        with self.assertRaisesRegex(
            ValueError,
            re.escape(f"Received: type(data)={type(test_text_invalid)}."),
        ):
            compute_sentence_piece_proto(test_text_invalid, 10)

    def test_lowercase(self):
        inputs = tf.data.Dataset.from_tensor_slices(["Drifting Along"])
        proto = compute_sentence_piece_proto(
            inputs, vocabulary_size=15, lowercase=True
        )
        tokenizer = SentencePieceTokenizer(proto=proto)
        output = inputs.map(tokenizer).take(1).get_single_element()
        expected_output = [4, 8, 12, 5, 9, 14, 5, 6, 13, 4, 7, 10, 11, 6, 13]
        self.assertAllEqual(expected_output, output)

    def test_proto_output_file(self):
        inputs = tf.data.Dataset.from_tensor_slices(["Drifting Along"])
        compute_sentence_piece_proto(
            inputs,
            vocabulary_size=15,
            proto_output_file=os.path.join(self.get_temp_dir(), "model.spm"),
        )
        tokenizer = SentencePieceTokenizer(
            proto=os.path.join(self.get_temp_dir(), "model.spm")
        )
        output = inputs.map(tokenizer).take(1).get_single_element()
        expected_output = [4, 8, 12, 5, 9, 14, 5, 6, 13, 4, 7, 10, 11, 6, 13]
        self.assertAllEqual(expected_output, output)
