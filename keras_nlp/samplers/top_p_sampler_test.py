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
"""Tests for Top-P sampler."""

import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.samplers.top_p_sampler import TopPSampler


class TopPSamplerTest(tf.test.TestCase, parameterized.TestCase):
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
        self.sampler = TopPSampler(p=0.1)

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
        def token_probability_fn(inputs, mask):
            batch_size, seq_length = tf.shape(inputs)[0], tf.shape(inputs)[1]
            prob = tf.constant([[[0.0, 0.0, 0.0, 1.0]]])
            return tf.tile(prob, [batch_size, seq_length, 1])

        inputs = tf.ragged.constant([[1], [2, 1, 2]])
        outputs = self.sampler(
            inputs,
            token_probability_fn,
            max_length=5,
            from_logits=False,
        )
        self.assertEqual(outputs.shape, [2, 5])

    @parameterized.named_parameters(
        ("xla_graph", True, False),
        ("non_xla_graph", False, False),
        ("eager", False, True),
    )
    def test_assert_probability_distribution_generation_is_correct(
        self, jit_compile, run_eagerly
    ):
        def token_probability_fn(inputs, mask):
            batch_size, seq_length = tf.shape(inputs)[0], tf.shape(inputs)[1]
            prob = tf.constant([[[0.0, 0.0, 0.0, 1.0]]])
            return tf.tile(prob, [batch_size, seq_length, 1])

        batch_size = 10
        inputs = 3 * tf.ones([batch_size, 1], dtype=tf.int32)
        max_length = 3

        outputs_count = np.array([0, 0, 0, 0])
        tf.random.set_seed(42)
        sampler = TopPSampler(
            p=0.1,
            seed=42,
            run_eagerly=jit_compile,
            jit_compile=run_eagerly,
        )
        for _ in range(8):
            outputs = sampler(
                inputs,
                token_probability_fn,
                max_length=max_length,
                from_logits=False,
            )
            flatten_predictions = tf.reshape(outputs[:, 1:], [-1])
            for pred in flatten_predictions:
                outputs_count[pred] += 1
        self.assertAllClose(
            outputs_count / np.sum(outputs_count),
            [0.0, 0.0, 0.0, 1.0],
            rtol=0.2,
        )

    def test_only_choose_from_top_p_tokens(self):
        # Test that there are only the top-p tokens in the output.
        def token_probability_fn(inputs, mask):
            batch_size, seq_length = tf.shape(inputs)[0], tf.shape(inputs)[1]
            prob = tf.constant([[[0.4, 0.3, 0.2, 0.1]]])
            return tf.tile(prob, [batch_size, seq_length, 1])

        # Test that it only samples from top-p tokens.
        for i, p in enumerate([0.399, 0.699, 0.899]):
            inputs = tf.constant([[0, 0], [0, 0]])
            sampler = TopPSampler(p=p)
            for _ in range(10):
                outputs = sampler(
                    inputs,
                    token_probability_fn,
                    max_length=5,
                    from_logits=False,
                )
                self.assertAllEqual(outputs <= i, tf.ones_like(outputs))

    def test_end_token_id(self):
        def token_probability_fn(inputs, mask):
            batch_size, seq_length = tf.shape(inputs)[0], tf.shape(inputs)[1]
            prob = tf.constant([[[0.0, 0.0, 0.0, 1.0]]])
            return tf.tile(prob, [batch_size, seq_length, 1])

        tf.random.set_seed(42)
        sampler = TopPSampler(p=0.1, seed=42)
        max_length = 4
        inputs = tf.constant([[0, 1], [1, 2]])
        outputs = sampler(
            inputs,
            token_probability_fn,
            max_length=max_length,
            end_token_id=2,
            from_logits=False,
        )
        # end_token in prompt does not trigger truncation.
        expected_outputs = tf.ragged.constant([[0, 1, 3, 3], [1, 2, 3, 3]])
        self.assertAllEqual(outputs, expected_outputs)

        outputs = sampler(
            inputs,
            token_probability_fn,
            max_length=max_length,
            end_token_id=3,
            from_logits=False,
        )
        # Generated end_token will be truncated.
        expected_outputs = tf.ragged.constant([[0, 1], [1, 2]])
        self.assertAllEqual(outputs, expected_outputs)
