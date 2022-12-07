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

"""Tests for RoBERTa preprocessing layers."""

import os

import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models.roberta.roberta_preprocessor import RobertaPreprocessor
from keras_nlp.models.roberta.roberta_preprocessor import RobertaTokenizer


class RobertaTokenizerTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        vocab = {
            "<s>": 0,
            "<pad>": 1,
            "</s>": 2,
            "Ġair": 3,
            "plane": 4,
            "Ġat": 5,
            "port": 6,
            "Ġkoh": 7,
            "li": 8,
            "Ġis": 9,
            "Ġthe": 10,
            "Ġbest": 11,
        }

        merges = ["Ġ a", "Ġ t", "Ġ k", "Ġ i", "Ġ b", "Ġa i", "p l", "n e"]
        merges += ["Ġa t", "p o", "r t", "o h", "l i", "Ġi s", "Ġb e", "s t"]
        merges += ["Ġt h", "Ġai r", "pl a", "Ġk oh", "Ġth e", "Ġbe st", "po rt"]
        merges += ["pla ne"]

        self.tokenizer = RobertaTokenizer(vocabulary=vocab, merges=merges)

    def test_tokenize(self):
        input_data = " airplane at airport"
        output = self.tokenizer(input_data)
        self.assertAllEqual(output, [3, 4, 5, 3, 6])

    def test_tokenize_batch(self):
        input_data = tf.constant([" airplane at airport", " kohli is the best"])
        output = self.tokenizer(input_data)
        self.assertAllEqual(output, [[3, 4, 5, 3, 6], [7, 8, 9, 10, 11]])

    def test_detokenize(self):
        input_tokens = [[3, 4, 5, 3, 6]]
        output = self.tokenizer.detokenize(input_tokens)
        self.assertAllEqual(output, [" airplane at airport"])

    def test_vocabulary_size(self):
        self.assertEqual(self.tokenizer.vocabulary_size(), 12)

    @parameterized.named_parameters(
        ("save_format_tf", "tf"), ("save_format_h5", "h5")
    )
    def test_saving_model(self, save_format):
        input_data = tf.constant([" airplane at airport"])

        inputs = keras.Input(dtype="string", shape=())
        outputs = self.tokenizer(inputs)
        model = keras.Model(inputs, outputs)

        path = os.path.join(self.get_temp_dir(), "model")
        model.save(path, save_format=save_format)

        restored_model = keras.models.load_model(path)
        self.assertAllEqual(
            model(input_data),
            restored_model(input_data),
        )


class RobertaPreprocessorTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        vocab = {
            "<s>": 0,
            "<pad>": 1,
            "</s>": 2,
            "Ġair": 3,
            "plane": 4,
            "Ġat": 5,
            "port": 6,
            "Ġkoh": 7,
            "li": 8,
            "Ġis": 9,
            "Ġthe": 10,
            "Ġbest": 11,
        }

        merges = ["Ġ a", "Ġ t", "Ġ k", "Ġ i", "Ġ b", "Ġa i", "p l", "n e"]
        merges += ["Ġa t", "p o", "r t", "o h", "l i", "Ġi s", "Ġb e", "s t"]
        merges += ["Ġt h", "Ġai r", "pl a", "Ġk oh", "Ġth e", "Ġbe st", "po rt"]
        merges += ["pla ne"]

        self.preprocessor = RobertaPreprocessor(
            tokenizer=RobertaTokenizer(
                vocabulary=vocab,
                merges=merges,
            ),
            sequence_length=12,
        )

    def test_tokenize(self):
        input_data = [" airplane at airport"]

        output = self.preprocessor(input_data)
        self.assertAllEqual(
            output["token_ids"], [0, 3, 4, 5, 3, 6, 2, 1, 1, 1, 1, 1]
        )
        self.assertAllEqual(
            output["padding_mask"], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        )

    def test_tokenize_batch(self):
        input_data = tf.constant(
            [
                " airplane at airport",
                " airplane at airport",
                " airplane at airport",
                " airplane at airport",
            ]
        )

        output = self.preprocessor(input_data)
        self.assertAllEqual(
            output["token_ids"],
            [[0, 3, 4, 5, 3, 6, 2, 1, 1, 1, 1, 1]] * 4,
        )

        self.assertAllEqual(
            output["padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]] * 4
        )

    def test_tokenize_multiple_sentences(self):
        sentence_one = " airplane at airport"
        sentence_two = " kohli is the best"

        output = self.preprocessor((sentence_one, sentence_two))
        self.assertAllEqual(
            output["token_ids"], [0, 3, 4, 5, 3, 2, 2, 7, 8, 9, 10, 2]
        )
        self.assertAllEqual(
            output["padding_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        )

    def test_tokenize_multiple_batched_sentences(self):
        sentence_one = tf.constant(
            [
                " airplane at airport",
                " airplane at airport",
                " airplane at airport",
                " airplane at airport",
            ]
        )
        sentence_two = tf.constant(
            [
                " kohli is the best",
                " kohli is the best",
                " kohli is the best",
                " kohli is the best",
            ]
        )

        output = self.preprocessor((sentence_one, sentence_two))
        self.assertAllEqual(
            output["token_ids"],
            [[0, 3, 4, 5, 3, 2, 2, 7, 8, 9, 10, 2]] * 4,
        )
        self.assertAllEqual(
            output["padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]] * 4
        )

    def test_detokenize(self):
        input_tokens = [0, 3, 4, 5, 3, 6, 2, 1, 1, 1, 1, 1]
        output = self.preprocessor.tokenizer.detokenize(input_tokens)
        self.assertEqual(
            output, "<s> airplane at airport</s><pad><pad><pad><pad><pad>"
        )

    @parameterized.named_parameters(
        ("save_format_tf", "tf"), ("save_format_h5", "h5")
    )
    def test_saving_model(self, save_format):
        input_data = tf.constant([" airplane at airport"])

        inputs = keras.Input(dtype="string", shape=())
        outputs = self.preprocessor(inputs)
        model = keras.Model(inputs, outputs)

        path = os.path.join(self.get_temp_dir(), "model")
        model.save(path, save_format=save_format)

        restored_model = keras.models.load_model(path)
        self.assertAllEqual(
            model(input_data)["token_ids"],
            restored_model(input_data)["token_ids"],
        )
