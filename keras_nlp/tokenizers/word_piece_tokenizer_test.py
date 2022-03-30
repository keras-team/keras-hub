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

import os

import tensorflow as tf
from tensorflow import keras

from keras_nlp.tokenizers.word_piece_tokenizer import WordPieceTokenizer


class WordPieceTokenizerTest(tf.test.TestCase):
    def test_tokenize(self):
        input_data = ["the quick brown fox."]
        vocab_data = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
        tokenizer = WordPieceTokenizer(vocabulary=vocab_data)
        call_output = tokenizer(input_data)
        tokenize_output = tokenizer.tokenize(input_data)
        self.assertIsInstance(call_output, tf.RaggedTensor)
        self.assertAllEqual(call_output, [[1, 2, 3, 4, 5, 6, 7]])
        self.assertAllEqual(tokenize_output, [[1, 2, 3, 4, 5, 6, 7]])

    def test_dense_output(self):
        input_data = ["the quick brown fox."]
        vocab_data = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
        tokenizer = WordPieceTokenizer(
            vocabulary=vocab_data, sequence_length=10
        )
        call_output = tokenizer(input_data)
        self.assertIsInstance(call_output, tf.Tensor)
        self.assertAllEqual(call_output, [[1, 2, 3, 4, 5, 6, 7, 0, 0, 0]])

    def test_string_tokenize(self):
        input_data = ["the quick brown fox"]
        vocab_data = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox"]
        tokenizer = WordPieceTokenizer(vocabulary=vocab_data, dtype="string")
        call_output = tokenizer(input_data)
        self.assertAllEqual(
            call_output,
            tf.ragged.constant([["the", "qu", "##ick", "br", "##own", "fox"]]),
        )

    def test_detokenize(self):
        input_data = [[1, 2, 3, 4, 5, 6]]
        vocab_data = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox"]
        tokenizer = WordPieceTokenizer(vocabulary=vocab_data)
        detokenize_output = tokenizer.detokenize(input_data)
        self.assertAllEqual(detokenize_output, ["the quick brown fox"])

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
            tf.ragged.constant([["qu", "@@ick", "br", "@@own", "@UNK@"]]),
        )

    def test_lowercase(self):
        input_data = ["the QUicK brOWN FOX"]
        vocab_data = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox"]
        tokenizer = WordPieceTokenizer(vocabulary=vocab_data)
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
        tokenizer = WordPieceTokenizer(vocabulary=vocab_data)
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

    def test_custom_spliting(self):
        input_data = ["a,e,i,o,u"]
        vocab_data = ["[UNK]", "a", "e", "i", "o", "u", ","]
        # Don't keep commas.
        tokenizer = WordPieceTokenizer(
            vocabulary=vocab_data, split_pattern=",", keep_pattern=""
        )
        call_output = tokenizer(input_data)
        self.assertAllEqual(call_output, [[1, 2, 3, 4, 5]])
        # Keep commas.
        tokenizer = WordPieceTokenizer(
            vocabulary=vocab_data, split_pattern=",", keep_pattern=","
        )
        call_output = tokenizer(input_data)
        self.assertAllEqual(call_output, [[1, 6, 2, 6, 3, 6, 4, 6, 5]])

    def test_no_spliting(self):
        input_data = ["t o k e n", "m i s s i n g", "t o k e n"]
        vocab_data = ["[UNK]", "t o k e n"]
        tokenizer = WordPieceTokenizer(
            vocabulary=vocab_data, split_pattern=None
        )
        call_output = tokenizer(input_data)
        self.assertAllEqual(call_output, [1, 0, 1])

    def test_word_piece_only(self):
        input_data = ["the", "quíck", "Brówn", "Fóx"]
        vocab_data = ["[UNK]", "the", "qu", "##íck", "Br", "##ówn", "Fóx"]
        tokenizer = WordPieceTokenizer(
            vocabulary=vocab_data,
            lowercase=False,
            strip_accents=False,
            split_pattern=None,
        )
        call_output = tokenizer(input_data)
        self.assertAllEqual(call_output, [1, 2, 3, 4, 5, 6])

    def test_functional_model(self):
        input_data = tf.constant(["the quick brown fox"])
        vocab_data = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox"]
        tokenizer = WordPieceTokenizer(vocabulary=vocab_data)
        inputs = keras.Input(dtype="string", shape=())
        outputs = tokenizer.detokenize(tokenizer.tokenize(inputs))
        model = keras.Model(inputs, outputs)
        model_output = model(input_data)
        self.assertAllEqual(model_output, ["the quick brown fox"])

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
        self.assertAllEqual(
            original_tokenizer(input_data),
            cloned_tokenizer(input_data),
        )

    def test_saving(self):
        input_data = tf.constant(["quick brOWN whale"])
        vocab_data = ["@UNK@", "qu", "@@ick", "br", "@@OWN", "fox"]
        tokenizer = WordPieceTokenizer(
            vocabulary=vocab_data,
            lowercase=False,
            oov_token="@UNK@",
            suffix_indicator="@@",
            dtype="string",
        )
        inputs = keras.Input(dtype="string", shape=())
        outputs = tokenizer(inputs)
        model = keras.Model(inputs, outputs)
        model.save(self.get_temp_dir())
        restored_model = keras.models.load_model(self.get_temp_dir())
        self.assertAllEqual(
            model(input_data),
            restored_model(input_data),
        )
