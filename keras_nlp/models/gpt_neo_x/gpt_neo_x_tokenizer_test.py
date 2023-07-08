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

"""Tests for GPT-2 preprocessing layers."""
import os

import pytest
import tensorflow as tf

from keras_nlp.backend import keras
from keras_nlp.models.gpt_neo_x.gpt_neo_x_tokenizer import GPTNeoXTokenizer
from keras_nlp.tests.test_case import TestCase


class GPTNeoXTokenizerTest(TestCase):
    def setUp(self):
        self.vocab = {
            "<|endoftext|>": 0,
            "Ġair": 1,
            "plane": 2,
            "Ġat": 3,
            "port": 4,
            "Ġkoh": 5,
            "li": 6,
            "Ġis": 7,
            "Ġthe": 8,
            "Ġbest": 9,
        }
        self.merges = ["Ġ a", "Ġ t", "Ġ k", "Ġ i", "Ġ b", "Ġa i", "p l", "n e"]
        self.merges += [
            "Ġa t",
            "p o",
            "r t",
            "o h",
            "l i",
            "Ġi s",
            "Ġb e",
            "s t",
        ]
        self.merges += [
            "Ġt h",
            "Ġai r",
            "pl a",
            "Ġk oh",
            "Ġth e",
            "Ġbe st",
            "po rt",
        ]
        self.merges += ["pla ne"]

        self.tokenizer = GPTNeoXTokenizer(
            vocabulary=self.vocab, merges=self.merges
        )

    def test_tokenize(self):
        input_data = " airplane at airport"
        output = self.tokenizer(input_data)
        self.assertAllEqual(output, [1, 2, 3, 1, 4])

    def test_tokenize_end_token(self):
        input_data = " airplane at airport<|endoftext|>"
        output = self.tokenizer(input_data)
        self.assertAllEqual(output, [1, 2, 3, 1, 4, 0])

    def test_tokenize_batch(self):
        input_data = [" airplane at airport", " kohli is the best"]
        output = self.tokenizer(input_data)
        self.assertAllEqual(output, [[1, 2, 3, 1, 4], [5, 6, 7, 8, 9]])

    def test_detokenize(self):
        input_tokens = [1, 2, 3, 1, 4]
        output = self.tokenizer.detokenize(input_tokens)
        self.assertEqual(output, " airplane at airport")

    def test_vocabulary_size(self):
        self.assertEqual(self.tokenizer.vocabulary_size(), 10)

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            GPTNeoXTokenizer(vocabulary=["a", "b", "c"], merges=[])

    def test_serialization(self):
        config = keras.saving.serialize_keras_object(self.tokenizer)
        new_tokenizer = keras.saving.deserialize_keras_object(config)
        self.assertEqual(
            new_tokenizer.get_config(),
            self.tokenizer.get_config(),
        )

    @pytest.mark.large
    @pytest.mark.tf_only
    def test_saved_model(self):
        input_data = tf.constant([" airplane at airport"])

        inputs = keras.Input(dtype="string", shape=())
        outputs = self.tokenizer(inputs)
        model = keras.Model(inputs, outputs)

        path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(path, save_format="keras_v3")

        restored_model = keras.models.load_model(path)
        self.assertAllEqual(
            model(input_data),
            restored_model(input_data),
        )
