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

"""Tests for GPT-2 preprocessing layers."""

import os

import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models.gpt2.gpt2_tokenizer import GPT2Tokenizer


class GPT2TokenizerTest(tf.test.TestCase, parameterized.TestCase):
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

        merges = ["Ġ a", "Ġ t", "Ġ k", "Ġ i", "Ġ b", "Ġa i", "p l", "n e"]
        merges += ["Ġa t", "p o", "r t", "o h", "l i", "Ġi s", "Ġb e", "s t"]
        merges += ["Ġt h", "Ġai r", "pl a", "Ġk oh", "Ġth e", "Ġbe st", "po rt"]
        merges += ["pla ne"]
        self.merges = merges

        self.tokenizer = GPT2Tokenizer(
            vocabulary=self.vocab, merges=self.merges
        )

    def test_tokenize(self):
        input_data = " airplane at airport"
        output = self.tokenizer(input_data)
        self.assertAllEqual(output, [1, 2, 3, 1, 4])

    def test_tokenize_batch(self):
        input_data = tf.constant([" airplane at airport", " kohli is the best"])
        output = self.tokenizer(input_data)
        self.assertAllEqual(output, [[1, 2, 3, 1, 4], [5, 6, 7, 8, 9]])

    def test_detokenize(self):
        input_tokens = [1, 2, 3, 1, 4]
        output = self.tokenizer.detokenize(input_tokens)
        self.assertEqual(output, " airplane at airport")

    def test_vocabulary_size(self):
        self.assertEqual(self.tokenizer.vocabulary_size(), 10)

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saved_model(self, save_format, filename):
        input_data = tf.constant([" airplane at airport"])

        inputs = keras.Input(dtype="string", shape=())
        outputs = self.tokenizer(inputs)
        model = keras.Model(inputs, outputs)

        path = os.path.join(self.get_temp_dir(), filename)
        model.save(path, save_format=save_format)

        restored_model = keras.models.load_model(path)
        self.assertAllEqual(
            model(input_data),
            restored_model(input_data),
        )
