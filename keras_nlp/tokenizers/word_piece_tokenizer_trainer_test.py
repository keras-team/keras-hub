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

from keras_nlp.tokenizers.word_piece_tokenizer_trainer import (
    compute_word_piece_vocabulary,
)


class WordPieceTokenizerTrainerTest(tf.test.TestCase):
    def test_dataset_input(self):
        test_text = ["baa maa caa saa aaa"]
        test_output = ["a", "b", "c", "m", "s", "##aa", "##a", "##b"]
        data = tf.data.Dataset.from_tensor_slices(test_text)
        vocab = compute_word_piece_vocabulary(data, 8, reserved_tokens=[])
        self.assertAllEqual(vocab, test_output)

    def test_filenames_input(self):
        test_text = "baa maa caa saa aaa"
        input_file = os.path.join(self.get_temp_dir(), "test.txt")
        with open(input_file, "w+") as f:
            f.write(test_text + "\n")
        test_output = ["a", "b", "c", "m", "s", "##aa", "##a", "##b"]
        vocab = compute_word_piece_vocabulary(
            [input_file],
            8,
            reserved_tokens=[],
        )
        self.assertAllEqual(vocab, test_output)

    def test_filenames_without_split(self):
        test_text = "baa maa caa saa aaa"
        input_file = os.path.join(self.get_temp_dir(), "test.txt")
        with open(input_file, "w+") as f:
            f.write(test_text + "\n")

        with self.assertRaisesRegex(
            ValueError,
            "When learning a vocab from files, `split` must be `True`. "
            "To compute a vocabulary with custom split rules, load your "
            "data as a dataset, split it, and pass it to "
            r"`compute_word_piece_vocabulary\(\)` with split=False.",
        ):
            compute_word_piece_vocabulary(["test.txt"], 10, split=False)

    def test_invalid_input(self):
        test_text_invalid = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4])
        with self.assertRaisesRegex(
            ValueError,
            "The dataset elements in `data` must have string dtype. "
            "Received: <dtype: 'int32'>.",
        ):
            compute_word_piece_vocabulary(test_text_invalid, 10)
        with self.assertRaisesRegex(
            ValueError,
            "The `data` argument must be either `tf.data.Dataset` or `list`. "
            "Received: <class 'int'>.",
        ):
            compute_word_piece_vocabulary(4, 4)

    def test_lowercase(self):
        test_text = tf.data.Dataset.from_tensor_slices(["BaA Maa Caa Saa AAa"])
        test_output = ["a", "b", "c", "m", "s", "##aa", "##a", "##b"]

        vocab = compute_word_piece_vocabulary(
            test_text, 8, lowercase=True, reserved_tokens=[]
        )
        self.assertAllEqual(vocab, test_output)

    def test_skip_lowercase(self):
        test_text = tf.data.Dataset.from_tensor_slices(["BAA MAA CAA SAA AAA"])
        test_output = ["A", "B", "C", "M", "S", "##AA", "##A", "##B"]

        vocab = compute_word_piece_vocabulary(
            test_text, 8, lowercase=False, reserved_tokens=[]
        )
        self.assertAllEqual(vocab, test_output)

    def test_split(self):
        test_text = tf.data.Dataset.from_tensor_slices(
            ["This string: would be split up."]
        )
        test_text_split = tf.data.Dataset.from_tensor_slices(
            ["This", "string", ":", "would", "be", "split", "up", "."]
        )
        output_vocab_1 = compute_word_piece_vocabulary(
            test_text, 20, split=True, lowercase=False, strip_accents=False
        )
        output_vocab_2 = compute_word_piece_vocabulary(
            test_text_split,
            20,
            split=False,
            lowercase=False,
            strip_accents=False,
        )
        self.assertAllEqual(output_vocab_1, output_vocab_2)

    def test_split_on_cjk(self):
        test_text = tf.data.Dataset.from_tensor_slices(["ah半推zz"])
        test_text_split = tf.data.Dataset.from_tensor_slices(
            ["ah", "半", "推", "zz"]
        )
        output_vocab_1 = compute_word_piece_vocabulary(
            test_text,
            4,
            split=True,
            split_on_cjk=True,
            lowercase=False,
            strip_accents=False,
        )
        output_vocab_2 = compute_word_piece_vocabulary(
            test_text_split,
            4,
            split=False,
            split_on_cjk=False,
            lowercase=False,
            strip_accents=False,
        )
        self.assertAllEqual(output_vocab_1, output_vocab_2)

    def test_skip_split(self):
        test_text = tf.data.Dataset.from_tensor_slices(
            [
                "This is a long line that isn't split up, and it exceeds maximum length."
            ]
        )
        # The token would be removed for being too long.
        vocab = compute_word_piece_vocabulary(
            test_text, 20, split=False, reserved_tokens=[]
        )
        self.assertAllEqual(vocab, [])

    def test_strip_accents(self):
        test_text = tf.data.Dataset.from_tensor_slices(
            ["áááá éááá íááá óááá úááá"]
        )
        output = ["a", "e", "i", "o", "u", "##aaa", "##a", "##e"]
        vocab = compute_word_piece_vocabulary(
            test_text, 8, strip_accents=True, reserved_tokens=[]
        )
        self.assertAllEqual(vocab, output)

    def test_skip_strip_accents(self):
        test_text = tf.data.Dataset.from_tensor_slices(
            ["áááá éááá íááá óááá úááá"]
        )
        output = ["á", "é", "í", "ó", "ú", "##ááá", "##á", "##é"]
        vocab = compute_word_piece_vocabulary(
            test_text, 8, strip_accents=False, reserved_tokens=[]
        )
        self.assertAllEqual(vocab, output)

    def test_output_file(self):
        test_text = tf.data.Dataset.from_tensor_slices(["BaA Maa Caa Saa AAa"])
        test_output = ["a", "b", "c", "m", "s", "##aa", "##a", "##b"]
        vocab_file = os.path.join(self.get_temp_dir(), "test.txt")
        compute_word_piece_vocabulary(
            test_text,
            8,
            vocab_file,
            lowercase=True,
            reserved_tokens=[],
        )
        vocab_from_file = []
        with open(vocab_file, "r") as f:
            for line in f:
                vocab_from_file.append(line.strip())
        self.assertAllEqual(vocab_from_file, test_output)

    def test_reserved_tokens(self):
        # This dummy text/token would be removed for being too long.
        test_text = tf.data.Dataset.from_tensor_slices(
            [
                "The learner requires at least one input here, but this should be removed."
            ]
        )
        output = ["token1", "token2", "token3", "token4"]
        vocab = compute_word_piece_vocabulary(
            test_text, 20, reserved_tokens=output, split=False
        )
        self.assertAllEqual(vocab, output)

    def test_suffix_indicator(self):
        test_text = tf.data.Dataset.from_tensor_slices(["baa maa caa saa aaa"])
        test_output = ["a", "b", "c", "m", "s", "@@aa", "@@a", "@@b"]
        vocab = compute_word_piece_vocabulary(
            test_text, 8, suffix_indicator="@@", reserved_tokens=[]
        )
        self.assertAllEqual(vocab, test_output)
