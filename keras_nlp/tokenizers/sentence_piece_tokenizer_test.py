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

import io
import os

import sentencepiece
import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.tokenizers.sentence_piece_tokenizer import SentencePieceTokenizer


class SentencePieceTokenizerTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        super().setUp()
        bytes_io = io.BytesIO()
        vocab_data = tf.data.Dataset.from_tensor_slices(
            ["the quick brown fox."]
        )
        sentencepiece.SentencePieceTrainer.train(
            sentence_iterator=vocab_data.as_numpy_iterator(),
            model_writer=bytes_io,
            vocab_size=7,
            model_type="WORD",
        )
        self.proto = bytes_io.getvalue()

    def test_tokenize(self):
        input_data = ["the quick brown fox."]
        tokenizer = SentencePieceTokenizer(
            proto=self.proto,
        )
        call_output = tokenizer(input_data)
        tokenize_output = tokenizer.tokenize(input_data)
        self.assertIsInstance(call_output, tf.RaggedTensor)
        self.assertAllEqual(call_output, [[6, 5, 3, 4]])
        self.assertAllEqual(tokenize_output, [[6, 5, 3, 4]])

    def test_scalar_tokenize(self):
        input_data = "the quick brown fox."
        tokenizer = SentencePieceTokenizer(
            proto=self.proto,
        )
        call_output = tokenizer(input_data)
        tokenize_output = tokenizer.tokenize(input_data)
        self.assertIsInstance(call_output, tf.Tensor)
        self.assertAllEqual(call_output, [6, 5, 3, 4])
        self.assertAllEqual(tokenize_output, [6, 5, 3, 4])

    def test_dense_output(self):
        input_data = ["the quick brown fox."]
        tokenizer = SentencePieceTokenizer(
            proto=self.proto,
            sequence_length=10,
        )
        output_data = tokenizer(input_data)
        self.assertIsInstance(output_data, tf.Tensor)
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
            tf.ragged.constant([["▁the", "▁quick", "▁brown", "▁fox."]]),
        )

    def test_detokenize(self):
        input_data = [[6, 5, 3, 4]]
        tokenizer = SentencePieceTokenizer(
            proto=self.proto,
        )
        output_data = tokenizer.detokenize(input_data)
        self.assertAllEqual(output_data, ["the quick brown fox."])

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

    def test_functional_model(self):
        input_data = tf.constant(["the quick brown fox."])
        tokenizer = SentencePieceTokenizer(
            proto=self.proto,
        )
        inputs = keras.Input(dtype="string", shape=())
        outputs = tokenizer.detokenize(tokenizer.tokenize(inputs))
        model = keras.Model(inputs, outputs)
        model_output = model(input_data)
        self.assertAllEqual(model_output, ["the quick brown fox."])

    def test_from_file(self):
        filepath = os.path.join(self.get_temp_dir(), "model.txt")
        input_data = ["the quick brown fox."]
        with tf.io.gfile.GFile(filepath, "wb") as file:
            file.write(self.proto)
        tokenizer = SentencePieceTokenizer(
            proto=filepath,
        )
        output_data = tokenizer(input_data)
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
        self.assertAllEqual(
            original_tokenizer(input_data),
            cloned_tokenizer(input_data),
        )

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saved_model(self, save_format, filename):
        filepath = os.path.join(self.get_temp_dir(), "model.txt")
        input_data = tf.constant(["the quick brown whale."])
        with tf.io.gfile.GFile(filepath, "wb") as file:
            file.write(self.proto)
        tokenizer = SentencePieceTokenizer(
            proto=filepath,
        )
        inputs = keras.Input(dtype="string", shape=())
        outputs = tokenizer(inputs)
        model = keras.Model(inputs, outputs)
        path = os.path.join(self.get_temp_dir(), filename)
        model.save(path, save_format=save_format)
        restored_model = keras.models.load_model(path)
        self.assertAllEqual(
            model(input_data),
            restored_model(input_data),
        )
