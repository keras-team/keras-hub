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
"""Tests for Word Piece Tokenizer Trainer."""

import os

import tensorflow as tf

from keras_nlp.tokenizers.word_piece_tokenizer_trainer import (
    compute_word_piece_vocabulary,
)


class WordPieceTokenizerTrainerTest(tf.test.TestCase):
    def test_dataset_input(self):
        test_text = ["bat mat cat sat pat."]
        test_output = [
            "[PAD]",
            "[CLS]",
            "[SEP]",
            "[UNK]",
            "[MASK]",
            ".",
            "a",
            "b",
            "c",
            "m",
            "p",
            "s",
            "t",
            "##at",
            "##.",
            "##a",
            "##b",
            "##c",
            "##m",
            "##p",
            "##s",
            "##t",
        ]
        data = tf.data.Dataset.from_tensor_slices(test_text)
        vocab = compute_word_piece_vocabulary(
            data,
            10,
        )
        self.assertAllEqual(set(vocab), set(test_output))

    def test_string_input(self):
        test_text = ["bat mat cat sat pat."]
        test_output = [
            "[PAD]",
            "[CLS]",
            "[SEP]",
            "[UNK]",
            "[MASK]",
            ".",
            "a",
            "b",
            "c",
            "m",
            "p",
            "s",
            "t",
            "##at",
            "##.",
            "##a",
            "##b",
            "##c",
            "##m",
            "##p",
            "##s",
            "##t",
        ]
        vocab = compute_word_piece_vocabulary(
            test_text,
            10,
        )
        self.assertAllEqual(set(vocab), set(test_output))

    def test_invalid_input(self):
        test_text_invalid = [1, 2, 3, 4]
        with self.assertRaisesRegex(
            ValueError,
            "The elements in `data` must be string type. "
            "Recieved: <class 'int'>.",
        ):
            compute_word_piece_vocabulary(test_text_invalid, 10)
        with self.assertRaisesRegex(
            ValueError,
            "The dataset elements in `data` must have string dtype. "
            "Recieved: <dtype: 'int32'>.",
        ):
            compute_word_piece_vocabulary(
                tf.data.Dataset.from_tensor_slices(test_text_invalid), 10
            )
        with self.assertRaisesRegex(
            ValueError,
            "The `data` argument must be either `tf.data.Dataset` or `list`. "
            "Recieved: <class 'int'>.",
        ):
            compute_word_piece_vocabulary(4, 4)

    def test_lowercase(self):
        test_text = ["Bat Mat Cat Sat Pat."]
        test_output = [
            "[PAD]",
            "[CLS]",
            "[SEP]",
            "[UNK]",
            "[MASK]",
            ".",
            "b",
            "c",
            "m",
            "p",
            "s",
            "a",
            "t",
            "##at",
            "##.",
            "##b",
            "##c",
            "##m",
            "##p",
            "##s",
            "##a",
            "##t",
        ]

        vocab = compute_word_piece_vocabulary(test_text, 20, lowercase=True)
        self.assertAllEqual(set(vocab), set(test_output))

    def test_skip_lowercase(self):
        test_text = ["Bat Mat Cat Sat Pat."]
        test_output = [
            "[PAD]",
            "[CLS]",
            "[SEP]",
            "[UNK]",
            "[MASK]",
            ".",
            "B",
            "C",
            "M",
            "P",
            "S",
            "a",
            "t",
            "##at",
            "##.",
            "##B",
            "##C",
            "##M",
            "##P",
            "##S",
            "##a",
            "##t",
        ]

        vocab = compute_word_piece_vocabulary(test_text, 20, lowercase=False)
        self.assertAllEqual(set(vocab), set(test_output))

    def test_split(self):
        test_text = [
            "This is a long line that would not be split up, since it exceeds maximum length."
        ]
        # The token would be removed for being too long.
        vocab = compute_word_piece_vocabulary(
            test_text, 20, split=False, reserved_tokens=[]
        )
        self.assertAllEqual(vocab, [])

    def test_skip_split(self):
        test_text = ["This string: would be split up."]
        test_text_split = [
            "This",
            "string",
            ":",
            "would",
            "be",
            "split",
            "up",
            ".",
        ]
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
        self.assertAllEqual(set(output_vocab_1), set(output_vocab_2))

    def test_strip_accents(self):
        test_text = ["áááá éááá íááá óááá úááá"]
        output = [
            "a",
            "e",
            "i",
            "o",
            "u",
            "##aaa",
            "##a",
            "##e",
            "##i",
            "##o",
            "##u",
        ]
        vocab = compute_word_piece_vocabulary(
            test_text, 20, strip_accents=True, reserved_tokens=[]
        )
        self.assertAllEqual(set(vocab), set(output))

    def test_skip_strip_accents(self):
        test_text = ["áááá éááá íááá óááá úááá"]
        output = [
            "á",
            "é",
            "í",
            "ó",
            "ú",
            "##ááá",
            "##á",
            "##é",
            "##í",
            "##ó",
            "##ú",
        ]
        vocab = compute_word_piece_vocabulary(
            test_text, 20, strip_accents=False, reserved_tokens=[]
        )
        self.assertAllEqual(set(vocab), set(output))

    def test_output_file(self):
        test_text = ["Bat Mat Cat Sat Pat."]
        test_output = [
            "[PAD]",
            "[CLS]",
            "[SEP]",
            "[UNK]",
            "[MASK]",
            ".",
            "b",
            "c",
            "m",
            "p",
            "s",
            "a",
            "t",
            "##at",
            "##.",
            "##b",
            "##c",
            "##m",
            "##p",
            "##s",
            "##a",
            "##t",
        ]

        vocab = compute_word_piece_vocabulary(
            test_text, 20, vocabulary_output_file="test.txt"
        )
        vocab_from_file = []
        with open("test.txt", "r") as f:
            for line in f:
                vocab_from_file.append(line.strip())
        self.assertAllEqual(set(vocab_from_file), set(test_output))
        self.assertAllEqual(set(vocab_from_file), set(vocab))
        os.remove("test.txt")

    def test_reserved_tokens(self):
        # This dummy text/token would be removed for being too long.
        test_text = [
            "The learner requires at least one input here, but this should be removed."
        ]
        output = ["token1", "token2", "token3", "token4"]
        vocab = compute_word_piece_vocabulary(
            test_text, 20, reserved_tokens=output, split=False
        )
        self.assertAllEqual(set(vocab), set(output))

    def test_suffix_indicator(self):
        test_text = ["bat mat cat sat pat."]
        test_output = [
            "[PAD]",
            "[CLS]",
            "[SEP]",
            "[UNK]",
            "[MASK]",
            ".",
            "a",
            "b",
            "c",
            "m",
            "p",
            "s",
            "t",
            "@@at",
            "@@.",
            "@@a",
            "@@b",
            "@@c",
            "@@m",
            "@@p",
            "@@s",
            "@@t",
        ]
        vocab = compute_word_piece_vocabulary(
            test_text, 10, suffix_indicator="@@"
        )
        self.assertAllEqual(set(vocab), set(test_output))
