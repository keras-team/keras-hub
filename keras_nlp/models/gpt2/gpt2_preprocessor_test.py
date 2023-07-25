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

"""Tests for GPT2 preprocessor layer."""

import tensorflow as tf

from keras_nlp.backend import keras
from keras_nlp.models.gpt2.gpt2_preprocessor import GPT2Preprocessor
from keras_nlp.models.gpt2.gpt2_tokenizer import GPT2Tokenizer
from keras_nlp.tests.test_case import TestCase


class GPT2PreprocessorTest(TestCase):
    def setUp(self):
        self.vocab = {
            "!": 0,
            "air": 1,
            "Ġair": 2,
            "plane": 3,
            "Ġat": 4,
            "port": 5,
            "<|endoftext|>": 6,
        }

        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]

        self.preprocessor = GPT2Preprocessor(
            tokenizer=GPT2Tokenizer(
                vocabulary=self.vocab,
                merges=self.merges,
            ),
            sequence_length=8,
        )

    def test_tokenize_strings(self):
        input_data = "airplane at airport"

        x = self.preprocessor(input_data)
        self.assertAllEqual(x["token_ids"], [6, 1, 3, 4, 2, 5, 6, 0])
        self.assertAllEqual(x["padding_mask"], [1, 1, 1, 1, 1, 1, 1, 0])

    def test_tokenize_list_of_strings(self):
        input_data = ["airplane at airport"] * 4

        x = self.preprocessor(input_data)
        self.assertAllEqual(x["token_ids"], [[6, 1, 3, 4, 2, 5, 6, 0]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 0]] * 4)

    def test_no_start_end_token(self):
        input_data = ["airplane at airport"] * 4

        preprocessor = GPT2Preprocessor(
            tokenizer=GPT2Tokenizer(
                vocabulary=self.vocab,
                merges=self.merges,
            ),
            sequence_length=8,
            add_start_token=False,
            add_end_token=False,
        )
        x = preprocessor(input_data)
        self.assertAllEqual(x["token_ids"], [[1, 3, 4, 2, 5, 0, 0, 0]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 1, 0, 0, 0]] * 4)

    def test_tokenize_labeled_batch(self):
        x = tf.constant(["airplane at airport"] * 4)
        y_in = tf.constant([1] * 4)
        sw_in = tf.constant([1.0] * 4)
        x, y, sw = self.preprocessor(x, y_in, sw_in)
        self.assertAllEqual(x["token_ids"], [[6, 1, 3, 4, 2, 5, 6, 0]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 0]] * 4)
        self.assertAllEqual(y, y_in)
        self.assertAllEqual(sw, sw_in)

    def test_tokenize_labeled_dataset(self):
        x = tf.constant(["airplane at airport"] * 4)
        ds = tf.data.Dataset.from_tensor_slices(x)
        ds = ds.map(self.preprocessor)
        x = ds.batch(4).take(1).get_single_element()
        self.assertAllEqual(x["token_ids"], [[6, 1, 3, 4, 2, 5, 6, 0]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 0]] * 4)

    def test_sequence_length_override(self):
        input_data = "airplane at airport"
        x = self.preprocessor(input_data, sequence_length=4)
        self.assertAllEqual(x["token_ids"], [6, 1, 3, 6])

    def test_serialization(self):
        config = keras.saving.serialize_keras_object(self.preprocessor)
        new_preprocessor = keras.saving.deserialize_keras_object(config)
        self.assertEqual(
            new_preprocessor.get_config(),
            self.preprocessor.get_config(),
        )
