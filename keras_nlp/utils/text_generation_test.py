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
"""Tests for Text Generation Utils."""


import tensorflow as tf
import numpy as np

from keras_nlp.utils.text_generation import greedy_search, random_sampling, top_k_sampling


class TextGenerationTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        vocab_size = 10
        feature_size = 16

        # Create a dummy model to predict the next token.
        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=[None]),
                tf.keras.layers.Embedding(
                    input_dim=vocab_size,
                    output_dim=feature_size,
                ),
                tf.keras.layers.Dense(vocab_size),
                tf.keras.layers.Softmax(),
            ]
        )

        def token_probability_fn(inputs):
            return model(inputs)[:, -1, :]

        self.token_probability_fn = token_probability_fn

    def test_generate_with_1d_prompt(self):
        inputs = tf.constant([1])
        outputs = greedy_search(self.token_probability_fn, inputs, max_length=5)
        self.assertEquals(outputs.shape, [5])
        outputs = random_sampling(self.token_probability_fn, inputs, max_length=5)
        self.assertEquals(outputs.shape, [5])
        outputs = top_k_sampling(self.token_probability_fn, inputs, max_length=5)
        self.assertEquals(outputs.shape, [5])

    def test_generate_with_2d_prompt(self):
        inputs = tf.constant([[1], [1]])
        outputs = greedy_search(self.token_probability_fn, inputs, max_length=5)
        self.assertEquals(outputs.shape, [2, 5])
        outputs = random_sampling(self.token_probability_fn, inputs, max_length=5)
        self.assertEquals(outputs.shape, [2, 5])
        outputs = top_k_sampling(self.token_probability_fn, inputs, max_length=5)
        self.assertEquals(outputs.shape, [2, 5])

    def test_generate_with_list_prompt(self):
        inputs = [[1], [1]]
        outputs = greedy_search(self.token_probability_fn, inputs, max_length=5)
        self.assertEquals(outputs.shape, [2, 5])
        outputs = random_sampling(self.token_probability_fn, inputs, max_length=5)
        self.assertEquals(outputs.shape, [2, 5])
        outputs = top_k_sampling(self.token_probability_fn, inputs, max_length=5)
        self.assertEquals(outputs.shape, [2, 5])

    def test_generate_with_ragged_prompt(self):
        inputs = tf.ragged.constant([[1], [2, 3]])
        with self.assertRaises(ValueError):
            greedy_search(self.token_probability_fn, inputs, max_length=5)
        with self.assertRaises(ValueError):
            random_sampling(self.token_probability_fn, inputs, max_length=5)
        with self.assertRaises(ValueError):
            top_k_sampling(self.token_probability_fn, inputs, max_length=5)

    def test_assert_generation_is_correct(self):
        def token_probability_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.constant([[0.01, 0.01, 0.08, 0.9]])
            return tf.repeat(prob, batch_size, axis=0)

        batch_size = 10
        inputs = 3 * tf.ones([batch_size, 1], dtype=tf.int32)
        max_length = 3
        outputs = greedy_search(
            token_probability_fn, inputs, max_length=max_length
        )
        self.assertAllEqual(
            outputs, 3 * tf.ones(shape=[batch_size, max_length])
        )
        outputs = random_sampling(
            token_probability_fn, inputs, max_length=max_length, seed=42
        )
        # Random sampling result with seed 42
        seeded_result = 3 * np.ones(shape=[batch_size, max_length])
        seeded_result[2][2] = 2
        seeded_result[5][2] = 2
        seeded_result[6][2] = 2
        seeded_result[7][1] = 2
        seeded_result[8][1] = 2
        seeded_result[8][2] = 2
        self.assertAllEqual(
            outputs, seeded_result
        )

        outputs = top_k_sampling(
            token_probability_fn, inputs, max_length=max_length, seed=42, k=2
        )
        seeded_result = 3 * np.ones(shape=[batch_size, max_length])
        self.assertAllEqual(
            outputs, seeded_result
        )

    def test_end_token_id(self):
        def token_probability_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.constant([[0.01, 0.01, 0.08, 0.9]])
            return tf.repeat(prob, batch_size, axis=0)

        max_length = 5
        inputs = tf.constant([[0, 1], [1, 2]])
        outputs = greedy_search(
            token_probability_fn,
            inputs,
            max_length=max_length,
            end_token_id=2,
            pad_token_id=0,
        )
        expected_outputs = tf.tile([[3], [0]], [1, max_length - 2])
        expected_outputs = tf.concat([inputs, expected_outputs], axis=1)
        self.assertAllEqual(outputs, expected_outputs)

        outputs = random_sampling(
            token_probability_fn,
            inputs,
            max_length=max_length,
            seed=42,
            end_token_id=2,
            pad_token_id=0,
        )
        # Random sampling result with seed 42
        expected_outputs = tf.tile([[3], [0]], [1, max_length - 2])
        expected_outputs = tf.concat([inputs, expected_outputs], axis=1)
        self.assertAllEqual(outputs, expected_outputs)

        outputs = top_k_sampling(
            token_probability_fn,
            inputs,
            max_length=max_length,
            k=2,
            seed=24,
            end_token_id=2,
            pad_token_id=0,
        )
        # Random sampling result with seed 24
        expected_outputs = tf.tile([[3], [0]], [1, max_length - 2])
        expected_outputs = tf.concat([inputs, expected_outputs], axis=1)
        self.assertAllEqual(outputs, expected_outputs)


