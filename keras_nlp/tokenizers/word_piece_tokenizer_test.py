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
from keras_nlp.tokenizers.word_piece_tokenizer import WordPieceTokenizer


class WordPieceTokenizerTest(TestCase):
    def test_tokenize(self):
        input_data = ["the quick brown fox."]
        vocab_data = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
        tokenizer = WordPieceTokenizer(vocabulary=vocab_data)
        call_output = tokenizer(input_data)
        tokenize_output = tokenizer.tokenize(input_data)
        self.assertAllEqual(call_output, [[1, 2, 3, 4, 5, 6, 7]])
        self.assertAllEqual(tokenize_output, [[1, 2, 3, 4, 5, 6, 7]])

    def test_dense_output(self):
        input_data = ["the quick brown fox."]
        vocab_data = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
        tokenizer = WordPieceTokenizer(
            vocabulary=vocab_data, sequence_length=10
        )
        call_output = tokenizer(input_data)
        self.assertAllEqual(call_output, [[1, 2, 3, 4, 5, 6, 7, 0, 0, 0]])

    def test_string_tokenize(self):
        input_data = ["the quick brown fox"]
        vocab_data = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox"]
        tokenizer = WordPieceTokenizer(vocabulary=vocab_data, dtype="string")
        call_output = tokenizer(input_data)
        self.assertAllEqual(
            call_output,
            [["the", "qu", "##ick", "br", "##own", "fox"]],
        )

    def test_detokenize(self):
        vocab_data = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox"]
        tokenizer = WordPieceTokenizer(vocabulary=vocab_data)
        outputs = tokenizer.detokenize([1, 2, 3, 4, 5, 6])
        self.assertAllEqual(outputs, "the quick brown fox")
        outputs = tokenizer.detokenize([[1, 2, 3, 4, 5, 6], [1, 6]])
        self.assertAllEqual(outputs, ["the quick brown fox", "the fox"])

    def test_accessors(self):
        vocab_data = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox"]
        tokenizer = WordPieceTokenizer(vocabulary=vocab_data)
        self.assertEqual(tokenizer.vocabulary_size(), 7)
        self.assertEqual(
            tokenizer.get_vocabulary(),
            ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox"],
        )
        self.assertEqual(tokenizer.id_to_token(0), "[UNK]")
        self.assertEqual(tokenizer.id_to_token(6), "fox")
        self.assertEqual(tokenizer.token_to_id("[UNK]"), 0)
        self.assertEqual(tokenizer.token_to_id("fox"), 6)

    def test_error_id_out_of_vocabulary(self):
        vocab_data = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
        tokenizer = WordPieceTokenizer(vocabulary=vocab_data)
        with self.assertRaises(ValueError):
            tokenizer.id_to_token(tokenizer.vocabulary_size())
        with self.assertRaises(ValueError):
            tokenizer.id_to_token(-1)

    def test_special_tokens(self):
        input_data = ["quick brown whale"]
        vocab_data = ["@UNK@", "qu", "@@ick", "br", "@@own", "fox"]
        tokenizer = WordPieceTokenizer(
            vocabulary=vocab_data,
            oov_token="@UNK@",
            suffix_indicator="@@",
            dtype="string",
        )
        call_output = tokenizer(input_data)
        self.assertAllEqual(
            call_output,
            [["qu", "@@ick", "br", "@@own", "@UNK@"]],
        )

    def test_cjk_tokens(self):
        input_data = ["ah半推zz"]
        vocab_data = ["[UNK]", "推", "敐", "乐", "半", "偷", "匕", "ah", "zz"]
        tokenizer = WordPieceTokenizer(vocabulary=vocab_data, dtype="string")
        call_output = tokenizer(input_data)
        self.assertAllEqual(
            call_output,
            [["ah", "半", "推", "zz"]],
        )

    def test_lowercase(self):
        input_data = ["the QUicK brOWN FOX"]
        vocab_data = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox"]
        tokenizer = WordPieceTokenizer(vocabulary=vocab_data, lowercase=True)
        call_output = tokenizer(input_data)
        self.assertAllEqual(call_output, [[1, 2, 3, 4, 5, 6]])

    def test_skip_lowercase(self):
        input_data = ["the QUicK brOWN FOX"]
        vocab_data = ["[UNK]", "the", "QU", "##icK", "br", "##OWN", "fox"]
        tokenizer = WordPieceTokenizer(vocabulary=vocab_data, lowercase=False)
        call_output = tokenizer(input_data)
        self.assertAllEqual(call_output, [[1, 2, 3, 4, 5, 0]])

    def test_strip_accents(self):
        input_data = ["á é í ó ú"]
        vocab_data = ["[UNK]", "a", "e", "i", "o", "u"]
        tokenizer = WordPieceTokenizer(
            vocabulary=vocab_data, strip_accents=True
        )
        call_output = tokenizer(input_data)
        self.assertAllEqual(call_output, [[1, 2, 3, 4, 5]])

    def test_skip_strip_accents(self):
        input_data = ["á é í ó ú"]
        vocab_data = ["[UNK]", "á", "é", "í", "ó", "ú"]
        tokenizer = WordPieceTokenizer(
            vocabulary=vocab_data, strip_accents=False
        )
        call_output = tokenizer(input_data)
        self.assertAllEqual(call_output, [[1, 2, 3, 4, 5]])

    def test_no_splitting(self):
        input_data = ["t o k e n", "m i s s i n g", "t o k e n"]
        vocab_data = ["[UNK]", "t o k e n"]
        tokenizer = WordPieceTokenizer(vocabulary=vocab_data, split=False)
        call_output = tokenizer(input_data)
        self.assertAllEqual(call_output, [1, 0, 1])

    def test_word_piece_only(self):
        input_data = ["the", "quíck", "Brówn", "Fóx"]
        vocab_data = ["[UNK]", "the", "qu", "##íck", "Br", "##ówn", "Fóx"]
        tokenizer = WordPieceTokenizer(
            vocabulary=vocab_data,
            lowercase=False,
            strip_accents=False,
            split=False,
        )
        call_output = tokenizer(input_data)
        self.assertAllEqual(call_output, [1, 2, 3, 4, 5, 6])

    def test_batching_ragged_tensors(self):
        tokenizer = WordPieceTokenizer(
            vocabulary=["[UNK]", "a", "b", "c", "d", "e", "f"]
        )
        dataset = tf.data.Dataset.from_tensor_slices(["a b c", "d e", "a f e"])
        dataset = dataset.map(tokenizer)
        dataset = dataset.apply(
            tf.data.experimental.dense_to_ragged_batch(batch_size=1)
        )
        element = dataset.take(1).get_single_element().numpy()
        self.assertAllEqual(element, [[1, 2, 3]])

    def test_from_file(self):
        vocab_path = os.path.join(self.get_temp_dir(), "vocab.txt")
        input_data = ["the quick brown fox."]
        vocab_data = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
        with tf.io.gfile.GFile(vocab_path, "w") as file:
            for piece in vocab_data:
                file.write(piece + "\n")
        tokenizer = WordPieceTokenizer(vocabulary=vocab_path)
        call_output = tokenizer(input_data)
        self.assertAllEqual(call_output, [[1, 2, 3, 4, 5, 6, 7]])

    def test_config(self):
        input_data = ["quick brOWN whale"]
        vocab_data = ["@UNK@", "qu", "@@ick", "br", "@@OWN", "fox"]
        original_tokenizer = WordPieceTokenizer(
            vocabulary=vocab_data,
            lowercase=False,
            oov_token="@UNK@",
            suffix_indicator="@@",
            dtype="string",
        )
        cloned_tokenizer = WordPieceTokenizer.from_config(
            original_tokenizer.get_config()
        )
        cloned_tokenizer.set_vocabulary(original_tokenizer.get_vocabulary())
        self.assertAllEqual(
            original_tokenizer(input_data),
            cloned_tokenizer(input_data),
        )

    def test_no_oov_token_in_vocabulary(self):
        vocab_data = ["qu", "@@ick", "br", "@@OWN", "fox"]
        with self.assertRaises(ValueError):
            WordPieceTokenizer(
                vocabulary=vocab_data,
            )

        vocab_data = ["@UNK@", "qu", "@@ick", "br", "@@OWN", "fox"]
        with self.assertRaises(ValueError):
            WordPieceTokenizer(
                vocabulary=vocab_data,
            )

        vocab_data = ["UNK", "qu", "@@ick", "br", "@@OWN", "fox"]
        with self.assertRaises(ValueError):
            WordPieceTokenizer(
                vocabulary=vocab_data,
            )

        with self.assertRaises(ValueError):
            WordPieceTokenizer(vocabulary=vocab_data, oov_token=None)
