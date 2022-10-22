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

from keras_nlp.models.roberta.roberta_preprocessing import RobertaPreprocessor

VOCAB_PATH = keras.utils.get_file(
    None,
    "https://storage.googleapis.com/keras-nlp/models/roberta_base/vocab.json",
)
MERGE_PATH = keras.utils.get_file(
    None,
    "https://storage.googleapis.com/keras-nlp/models/roberta_base/merges.txt",
)


class RobertaPreprocessorTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.preprocessor = RobertaPreprocessor(
            vocabulary=VOCAB_PATH,
            merges=MERGE_PATH,
            sequence_length=10,
        )

    def test_tokenize(self):
        input_data = ["the quick brown fox."]

        output = self.preprocessor(input_data)
        self.assertAllEqual(
            output["token_ids"], [0, 627, 2119, 6219, 23602, 4, 2, 1, 1, 1]
        )
        self.assertAllEqual(
            output["padding_mask"], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
        )

    def test_tokenize_batch(self):
        input_data = tf.constant(
            [
                "the quick brown fox.",
                "the quick brown fox.",
                "the quick brown fox.",
                "the quick brown fox.",
            ]
        )

        output = self.preprocessor(input_data)
        self.assertAllEqual(
            output["token_ids"],
            [[0, 627, 2119, 6219, 23602, 4, 2, 1, 1, 1]] * 4,
        )

        self.assertAllEqual(
            output["padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0]] * 4
        )

    def test_tokenize_multiple_sentences(self):
        sentence_one = "kohli is the best batsman"
        sentence_two = "bumrah is the best bowler"

        output = self.preprocessor((sentence_one, sentence_two))
        self.assertAllEqual(
            output["token_ids"], [0, 330, 2678, 3572, 2, 2, 30406, 9772, 16, 2]
        )
        self.assertAllEqual(
            output["padding_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        )

    def test_tokenize_multiple_batched_sentences(self):
        sentence_one = tf.constant(
            [
                "kohli is the best batsman",
                "kohli is the best batsman",
                "kohli is the best batsman",
                "kohli is the best batsman",
            ]
        )
        sentence_two = tf.constant(
            [
                "bumrah is the best bowler",
                "bumrah is the best bowler",
                "bumrah is the best bowler",
                "bumrah is the best bowler",
            ]
        )

        output = self.preprocessor((sentence_one, sentence_two))
        self.assertAllEqual(
            output["token_ids"],
            [[0, 330, 2678, 3572, 2, 2, 30406, 9772, 16, 2]] * 4,
        )
        self.assertAllEqual(
            output["padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]] * 4
        )

    def test_detokenize(self):
        input_tokens = [[627, 2119, 6219, 23602, 4]]
        output = self.preprocessor.tokenizer.detokenize(input_tokens)
        self.assertAllEqual(output, ["the quick brown fox."])

    def test_vocabulary_size(self):
        preprocessor = RobertaPreprocessor(
            vocabulary=VOCAB_PATH,
            merges=MERGE_PATH,
        )
        self.assertEqual(preprocessor.vocabulary_size(), 50265)

    @parameterized.named_parameters(
        ("save_format_tf", "tf"), ("save_format_h5", "h5")
    )
    def test_saving_model(self, save_format):
        input_data = tf.constant(["the quick brown fox."])

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
