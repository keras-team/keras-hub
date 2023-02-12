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
"""Tests for Greedy sampler."""

import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.samplers.greedy_sampler import GreedySampler


class GreedySamplerTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.vocab_size = 10
        self.feature_size = 16

        # Create a dummy model to predict the next token.
        model = keras.Sequential(
            [
                keras.Input(shape=[None]),
                keras.layers.Embedding(
                    input_dim=self.vocab_size,
                    output_dim=self.feature_size,
                ),
                keras.layers.Dense(self.vocab_size),
                keras.layers.Softmax(),
            ]
        )

        def token_probability_fn(inputs, mask):
            return model(inputs)

        self.token_probability_fn = token_probability_fn

        self.sampler = GreedySampler()

    def test_generate_with_1d_prompt(self):
        inputs = tf.constant([1])
        outputs = self.sampler(inputs, self.token_probability_fn, max_length=5)
        self.assertEqual(outputs.shape, [5])

    def test_generate_with_2d_prompt(self):
        inputs = tf.constant([[1], [1]])
        outputs = self.sampler(inputs, self.token_probability_fn, max_length=5)
        self.assertEqual(outputs.shape, [2, 5])

    def test_generate_with_list_prompt(self):
        inputs = [[1], [1]]
        outputs = self.sampler(inputs, self.token_probability_fn, max_length=5)
        self.assertEqual(outputs.shape, [2, 5])

    def test_generate_with_ragged_prompt(self):
        max_length = 5

        def token_probability_fn(inputs, mask):
            # Assert that user function is passed only dense tensors.
            self.assertIsInstance(inputs, tf.Tensor)
            prob = tf.constant([[[0.0, 0.0, 0.0, 1.0]]])
            return tf.repeat(tf.repeat(prob, 2, axis=0), max_length, axis=1)

        inputs = tf.ragged.constant([[1], [2, 1, 2]])
        outputs = self.sampler(inputs, token_probability_fn, max_length)
        self.assertEqual(outputs.shape, [2, 5])

    def test_assert_generation_is_correct(self):
        batch_size = 10
        max_length = 3

        def token_probability_fn(inputs, mask):
            prob = tf.constant([[[0.0, 0.0, 0.0, 1.0]]])
            return tf.repeat(
                tf.repeat(prob, batch_size, axis=0), max_length, axis=1
            )

        inputs = 3 * tf.ones([batch_size, 1], dtype=tf.int32)
        outputs = self.sampler(
            inputs, token_probability_fn, max_length=max_length
        )
        self.assertAllEqual(
            outputs, 3 * tf.ones(shape=[batch_size, max_length])
        )

    def test_end_token_id(self):
        def token_probability_fn(inputs, mask):
            batch_size = inputs.shape[0]
            prob = tf.constant([[[0.0, 0.0, 0.0, 1.0]]])
            return tf.repeat(
                tf.repeat(prob, batch_size, axis=0), max_length, axis=1
            )

        max_length = 4
        sampler = GreedySampler()
        inputs = tf.constant([[0, 1], [1, 2]])
        outputs = sampler(
            inputs,
            token_probability_fn,
            max_length=max_length,
            end_token_id=2,
        )
        # end_token in prompt does not trigger truncation.
        expected_outputs = tf.ragged.constant([[0, 1, 3, 3], [1, 2, 3, 3]])
        self.assertAllEqual(outputs, expected_outputs)

        outputs = sampler(
            inputs,
            token_probability_fn,
            max_length=max_length,
            end_token_id=3,
        )
        # Generated end_token will be truncated.
        expected_outputs = tf.ragged.constant([[0, 1], [1, 2]])
        self.assertAllEqual(outputs, expected_outputs)

    def test_compare_xla_noxla_results(self):
        inputs = [[1], [1]]
        xla_sampler = GreedySampler(jit_compile=True)
        outputs_xla = xla_sampler(
            inputs, self.token_probability_fn, max_length=5
        )

        xla_sampler = GreedySampler(jit_compile=False)
        outputs_no_xla = xla_sampler(
            inputs, self.token_probability_fn, max_length=5
        )

        self.assertAllEqual(outputs_xla, outputs_no_xla)
