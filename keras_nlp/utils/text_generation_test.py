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

import random

import numpy as np
import tensorflow as tf

from keras_nlp.utils.text_generation import beam_search
from keras_nlp.utils.text_generation import greedy_search
from keras_nlp.utils.text_generation import random_search
from keras_nlp.utils.text_generation import top_k_search
from keras_nlp.utils.text_generation import top_p_search


class GreedySearchTextGenerationTest(tf.test.TestCase):
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

    def test_generate_with_empty_prompt(self):
        inputs = tf.constant([])
        with self.assertRaises(ValueError):
            greedy_search(self.token_probability_fn, inputs, max_length=5)
        inputs = tf.constant([[]])
        with self.assertRaises(ValueError):
            greedy_search(self.token_probability_fn, inputs, max_length=5)

    def test_generate_with_1d_prompt(self):
        inputs = tf.constant([1])
        outputs = greedy_search(self.token_probability_fn, inputs, max_length=5)
        self.assertEquals(outputs.shape, [5])

    def test_generate_with_2d_prompt(self):
        inputs = tf.constant([[1], [1]])
        outputs = greedy_search(self.token_probability_fn, inputs, max_length=5)
        self.assertEquals(outputs.shape, [2, 5])

    def test_generate_with_list_prompt(self):
        inputs = [[1], [1]]
        outputs = greedy_search(self.token_probability_fn, inputs, max_length=5)
        self.assertEquals(outputs.shape, [2, 5])

    def test_generate_with_ragged_prompt(self):
        inputs = tf.ragged.constant([[1], [2, 3]])
        with self.assertRaises(ValueError):
            greedy_search(self.token_probability_fn, inputs, max_length=5)

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


class BeamSearchTextGenerationTest(tf.test.TestCase):
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

    def test_generate_with_empty_prompt(self):
        inputs = tf.constant([])
        with self.assertRaises(ValueError):
            beam_search(
                self.token_probability_fn, inputs, max_length=5, num_beams=5
            )
        inputs = tf.constant([[]])
        with self.assertRaises(ValueError):
            beam_search(
                self.token_probability_fn, inputs, max_length=5, num_beams=5
            )

    def test_generate_with_1d_prompt(self):
        inputs = tf.constant([1])
        outputs = beam_search(
            self.token_probability_fn,
            inputs,
            max_length=5,
            num_beams=5,
        )
        self.assertEquals(outputs.shape, [5])

    def test_generate_with_2d_prompt(self):
        inputs = tf.constant([[1], [1]])
        outputs = beam_search(
            self.token_probability_fn,
            inputs,
            max_length=5,
            num_beams=5,
        )
        self.assertEquals(outputs.shape, [2, 5])

    def test_generate_with_list_prompt(self):
        inputs = [[1], [1]]
        outputs = beam_search(
            self.token_probability_fn,
            inputs,
            max_length=5,
            num_beams=5,
        )
        self.assertEquals(outputs.shape, [2, 5])

    def test_generate_with_ragged_prompt(self):
        inputs = tf.ragged.constant([[1], [2, 3]])
        with self.assertRaises(ValueError):
            beam_search(
                self.token_probability_fn,
                inputs,
                max_length=5,
                num_beams=5,
            )

    def test_one_beam_generation(self):
        for i in range(50):
            inputs = tf.constant([random.randint(0, 9)])
            beam_output = beam_search(
                self.token_probability_fn,
                inputs,
                max_length=5,
                num_beams=1,
            )
            greedy_output = greedy_search(
                self.token_probability_fn,
                inputs,
                max_length=5,
            )
            self.assertAllEqual(beam_output, greedy_output)

    def test_multiple_beam_generation(self):
        def token_probability_fn(inputs):
            if inputs.shape[1] == 1:
                prob = tf.constant([[0.1, 0.2, 0.3, 0.4]])
            elif inputs[0][1] == 2:
                prob = tf.constant([[0.9, 0.08, 0.01, 0.01]])
            elif inputs[0][1] == 3:
                prob = tf.constant([[0.25, 0.25, 0.25, 0.25]])
            return prob

        inputs = tf.constant([[1]])
        beam_output = beam_search(
            token_probability_fn,
            inputs,
            max_length=3,
            num_beams=2,
        )
        self.assertAllEqual(
            beam_output, tf.constant([1, 2, 0], dtype=beam_output.dtype)
        )

    def test_assert_generation_is_correct(self):
        def token_probability_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.constant([[0.01, 0.01, 0.08, 0.9]])
            return tf.repeat(prob, batch_size, axis=0)

        batch_size = 10
        inputs = 3 * tf.ones([batch_size, 1], dtype=tf.int32)
        max_length = 3
        for i in range(1, 10):
            outputs = beam_search(
                token_probability_fn,
                inputs,
                max_length=max_length,
                num_beams=i,
            )
            self.assertAllEqual(
                outputs, 3 * tf.ones(shape=[batch_size, max_length])
            )

    def test_end_token_id(self):
        def token_probability_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.constant([[0.01, 0.01, 0.08, 0.9]])
            return tf.repeat(prob, batch_size, axis=0)

        max_length = 5
        inputs = tf.constant([[0, 1], [1, 2]])
        outputs = beam_search(
            token_probability_fn,
            inputs,
            max_length=max_length,
            num_beams=2,
            end_token_id=2,
            pad_token_id=0,
        )
        expected_outputs = tf.tile([[3], [0]], [1, max_length - 2])
        expected_outputs = tf.concat([inputs, expected_outputs], axis=1)
        self.assertAllEqual(outputs, expected_outputs)


class RandomSearchTextGenerationTest(tf.test.TestCase):
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

    def test_generate_with_empty_prompt(self):
        inputs = tf.constant([])
        with self.assertRaises(ValueError):
            random_search(self.token_probability_fn, inputs, max_length=5)
        inputs = tf.constant([[]])
        with self.assertRaises(ValueError):
            random_search(self.token_probability_fn, inputs, max_length=5)

    def test_generate_with_1d_prompt(self):
        inputs = tf.constant([1])
        outputs = random_search(self.token_probability_fn, inputs, max_length=5)
        self.assertEquals(outputs.shape, [5])

    def test_generate_with_2d_prompt(self):
        inputs = tf.constant([[1], [1]])
        outputs = random_search(self.token_probability_fn, inputs, max_length=5)
        self.assertEquals(outputs.shape, [2, 5])

    def test_generate_with_list_prompt(self):
        inputs = [[1], [1]]
        outputs = random_search(self.token_probability_fn, inputs, max_length=5)
        self.assertEquals(outputs.shape, [2, 5])

    def test_generate_with_ragged_prompt(self):
        inputs = tf.ragged.constant([[1], [2, 3]])
        with self.assertRaises(ValueError):
            random_search(self.token_probability_fn, inputs, max_length=5)

    def test_assert_seeded_generation_is_correct(self):
        def token_probability_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.constant([[0.01, 0.01, 0.08, 0.9]])
            return tf.repeat(prob, batch_size, axis=0)

        batch_size = 10
        inputs = 3 * tf.ones([batch_size, 1], dtype=tf.int32)
        max_length = 3
        tf.random.set_seed(42)
        outputs = random_search(
            token_probability_fn, inputs, max_length=max_length, seed=42
        )
        # Random sampling result with seed 42.
        seeded_result = 3 * np.ones(shape=[batch_size, max_length])
        self.assertAllEqual(outputs, seeded_result)

    def test_assert_probability_distribution_generation_is_correct(self):
        def token_probability_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.constant([[0.01, 0.01, 0.08, 0.9]])
            return tf.repeat(prob, batch_size, axis=0)

        batch_size = 10
        inputs = 3 * tf.ones([batch_size, 1], dtype=tf.int32)
        max_length = 3

        outputs_count = np.array([0, 0, 0, 0])
        tf.random.set_seed(42)
        for i in range(500):
            outputs = random_search(
                token_probability_fn, inputs, max_length=max_length, seed=42
            )
            flatten_predictions = tf.reshape(outputs[:, 1:], [-1])
            for pred in flatten_predictions:
                outputs_count[pred] += 1
        self.assertAllClose(
            outputs_count / np.sum(outputs_count),
            [0.01, 0.01, 0.08, 0.9],
            rtol=0.2,
        )

    def test_end_token_id(self):
        def token_probability_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.constant([[0.01, 0.01, 0.08, 0.9]])
            return tf.repeat(prob, batch_size, axis=0)

        max_length = 5
        inputs = tf.constant([[0, 1], [1, 2]])
        tf.random.set_seed(42)
        outputs = random_search(
            token_probability_fn,
            inputs,
            max_length=max_length,
            seed=42,
            end_token_id=2,
            pad_token_id=0,
        )
        # Random sampling result with seed 42.
        expected_outputs = tf.tile([[3], [0]], [1, max_length - 2])
        expected_outputs = tf.concat([inputs, expected_outputs], axis=1)
        self.assertAllEqual(outputs, expected_outputs)

    def test_from_logits(self):
        def token_logits_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.constant([[1.0, 2.0, 3, 0, 4.0]])
            return tf.repeat(prob, batch_size, axis=0)

        def token_probability_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.keras.activations.softmax(
                tf.constant([[1.0, 2.0, 3, 0, 4.0]])
            )
            return tf.repeat(prob, batch_size, axis=0)

        max_length = 5
        inputs = tf.constant([[0, 1], [1, 2]])
        tf.random.set_seed(42)
        output_logit = random_search(
            token_logits_fn,
            inputs,
            max_length=max_length,
            from_logits=True,
            seed=42,
        )
        tf.random.set_seed(42)
        output_probs = random_search(
            token_probability_fn,
            inputs,
            max_length=max_length,
            from_logits=False,
            seed=42,
        )
        self.assertAllEqual(output_logit, output_probs)


class TopKSearchTextGenerationTest(tf.test.TestCase):
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

    def test_generate_with_empty_prompt(self):
        inputs = tf.constant([])
        with self.assertRaises(ValueError):
            top_k_search(self.token_probability_fn, inputs, max_length=5, k=2)
        inputs = tf.constant([[]])
        with self.assertRaises(ValueError):
            top_k_search(self.token_probability_fn, inputs, max_length=5, k=2)

    def test_generate_with_1d_prompt(self):
        inputs = tf.constant([1])
        outputs = top_k_search(
            self.token_probability_fn, inputs, max_length=5, k=2
        )
        self.assertEquals(outputs.shape, [5])

    def test_generate_with_2d_prompt(self):
        inputs = tf.constant([[1], [1]])
        outputs = top_k_search(
            self.token_probability_fn, inputs, max_length=5, k=2
        )
        self.assertEquals(outputs.shape, [2, 5])

    def test_generate_with_list_prompt(self):
        inputs = [[1], [1]]
        outputs = top_k_search(
            self.token_probability_fn, inputs, max_length=5, k=2
        )
        self.assertEquals(outputs.shape, [2, 5])

    def test_generate_with_ragged_prompt(self):
        inputs = tf.ragged.constant([[1], [2, 3]])
        with self.assertRaises(ValueError):
            top_k_search(self.token_probability_fn, inputs, max_length=5, k=2)

    def test_assert_seeded_generation_is_correct(self):
        def token_probability_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.constant([[0.01, 0.01, 0.08, 0.9]])
            return tf.repeat(prob, batch_size, axis=0)

        batch_size = 10
        inputs = 3 * tf.ones([batch_size, 1], dtype=tf.int32)
        max_length = 3
        tf.random.set_seed(42)
        outputs = top_k_search(
            token_probability_fn, inputs, max_length=max_length, k=2, seed=42
        )
        # Top-k sampling result with seed 42.
        seeded_result = 3 * np.ones(shape=[batch_size, max_length])
        self.assertAllEqual(outputs, seeded_result)

    def test_assert_probability_distribution_generation_is_correct(self):
        def token_probability_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.constant([[0.1, 0.2, 0.3, 0.4]])
            return tf.repeat(prob, batch_size, axis=0)

        batch_size = 10
        inputs = 3 * tf.ones([batch_size, 1], dtype=tf.int32)
        max_length = 3

        outputs_count = np.array([0, 0, 0, 0])
        tf.random.set_seed(42)
        for i in range(500):
            outputs = top_k_search(
                token_probability_fn,
                inputs,
                max_length=max_length,
                k=2,
                seed=42,
            )
            flatten_predictions = tf.reshape(outputs[:, 1:], [-1])
            for pred in flatten_predictions:
                outputs_count[pred] += 1
        self.assertAllClose(
            outputs_count / np.sum(outputs_count),
            [0.0, 0.0, 0.429, 0.571],
            rtol=0.2,
        )

    def test_only_choose_from_top_k_tokens(self):
        # Test that there are only the top-k tokens in the output.
        def token_probability_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.constant([[0.4, 0.3, 0.2, 0.1]])
            return tf.repeat(prob, batch_size, axis=0)

        # Test that it only samples from top-k tokens.
        for k in [1, 2, 3]:
            inputs = tf.constant([[0, 0], [0, 0]])
            for _ in range(10):
                outputs = top_k_search(
                    token_probability_fn,
                    inputs,
                    max_length=5,
                    k=k,
                )
                self.assertAllEqual(outputs < k, tf.ones_like(outputs))

    def test_end_token_id(self):
        def token_probability_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.constant([[0.01, 0.01, 0.08, 0.9]])
            return tf.repeat(prob, batch_size, axis=0)

        max_length = 5
        inputs = tf.constant([[0, 1], [1, 2]])
        tf.random.set_seed(42)
        outputs = top_k_search(
            token_probability_fn,
            inputs,
            max_length=max_length,
            k=4,
            seed=42,
            end_token_id=2,
            pad_token_id=0,
        )
        # Top-k sampling result with seed 42.
        expected_outputs = tf.tile([[3], [0]], [1, max_length - 2])
        expected_outputs = tf.concat([inputs, expected_outputs], axis=1)
        self.assertAllEqual(outputs, expected_outputs)

    def test_from_logits(self):
        def token_logits_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.constant([[1.0, 2.0, 3.0, 4.0]])
            return tf.repeat(prob, batch_size, axis=0)

        def token_probability_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.keras.activations.softmax(
                tf.constant([[1.0, 2.0, 3.0, 4.0]])
            )
            return tf.repeat(prob, batch_size, axis=0)

        max_length = 5
        inputs = tf.constant([[0, 1], [1, 2]])
        tf.random.set_seed(42)
        output_logit = top_k_search(
            token_logits_fn,
            inputs,
            max_length=max_length,
            k=3,
            from_logits=True,
            seed=42,
        )
        tf.random.set_seed(42)
        output_probs = top_k_search(
            token_probability_fn,
            inputs,
            max_length=max_length,
            k=3,
            from_logits=False,
            seed=42,
        )
        self.assertAllEqual(output_logit, output_probs)


class TopPSearchTextGenerationTest(tf.test.TestCase):
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

    def test_generate_with_empty_prompt(self):
        inputs = tf.constant([])
        with self.assertRaises(ValueError):
            top_p_search(self.token_probability_fn, inputs, max_length=5, p=0.8)
        inputs = tf.constant([[]])
        with self.assertRaises(ValueError):
            top_p_search(self.token_probability_fn, inputs, max_length=5, p=0.8)

    def test_generate_with_1d_prompt(self):
        inputs = tf.constant([1])
        outputs = top_p_search(
            self.token_probability_fn, inputs, max_length=5, p=0.8
        )
        self.assertEquals(outputs.shape, [5])

    def test_generate_with_2d_prompt(self):
        inputs = tf.constant([[1], [1]])
        outputs = top_p_search(
            self.token_probability_fn, inputs, max_length=5, p=0.8
        )
        self.assertEquals(outputs.shape, [2, 5])

    def test_generate_with_list_prompt(self):
        inputs = [[1], [1]]
        outputs = top_p_search(
            self.token_probability_fn, inputs, max_length=5, p=0.8
        )
        self.assertEquals(outputs.shape, [2, 5])

    def test_generate_with_ragged_prompt(self):
        inputs = tf.ragged.constant([[1], [2, 3]])
        with self.assertRaises(ValueError):
            top_p_search(self.token_probability_fn, inputs, max_length=5, p=0.8)

    def test_assert_seeded_generation_is_correct(self):
        def token_probability_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.constant([[0.01, 0.01, 0.08, 0.9]])
            return tf.repeat(prob, batch_size, axis=0)

        batch_size = 10
        inputs = 3 * tf.ones([batch_size, 1], dtype=tf.int32)
        max_length = 3
        tf.random.set_seed(42)
        outputs = top_p_search(
            token_probability_fn, inputs, max_length=max_length, p=0.91, seed=42
        )
        # Top-p sampling result with seed 42.
        seeded_result = 3 * np.ones(shape=[batch_size, max_length])
        seeded_result[3][1] = 2
        seeded_result[7][1] = 2
        self.assertAllEqual(outputs, seeded_result)

    def test_assert_probability_distribution_generation_is_correct(self):
        def token_probability_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.constant([[0.1, 0.2, 0.3, 0.4]])
            return tf.repeat(prob, batch_size, axis=0)

        batch_size = 10
        inputs = 3 * tf.ones([batch_size, 1], dtype=tf.int32)
        max_length = 3

        outputs_count = np.array([0, 0, 0, 0])
        tf.random.set_seed(42)
        for i in range(500):
            outputs = top_p_search(
                token_probability_fn,
                inputs,
                max_length=max_length,
                p=0.6,
                seed=42,
            )
            flatten_predictions = tf.reshape(outputs[:, 1:], [-1])
            for pred in flatten_predictions:
                outputs_count[pred] += 1
        self.assertAllClose(
            outputs_count / np.sum(outputs_count),
            [0.0, 0.0, 0.429, 0.571],
            rtol=0.2,
        )

    def test_only_choose_from_top_p_tokens(self):
        def token_probability_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.constant([[0.4, 0.3, 0.2, 0.1]])
            return tf.repeat(prob, batch_size, axis=0)

        # Test that it only samples from tokens that sum up to p.
        for p, n in [(0.3, 1), (0.7, 2), (0.9, 3)]:
            inputs = tf.constant([[0, 0], [0, 0]])
            for _ in range(10):
                outputs = top_p_search(
                    token_probability_fn, inputs, max_length=5, p=p
                )
                self.assertAllEqual(outputs < n, tf.ones_like(outputs))

    def test_end_token_id(self):
        def token_probability_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.constant([[0.01, 0.01, 0.08, 0.9]])
            return tf.repeat(prob, batch_size, axis=0)

        max_length = 5
        inputs = tf.constant([[0, 1], [1, 2]])
        tf.random.set_seed(42)
        outputs = top_p_search(
            token_probability_fn,
            inputs,
            max_length=max_length,
            p=0.92,
            seed=1,
            end_token_id=2,
            pad_token_id=0,
        )
        # Top-p sampling result with seed 42.
        expected_outputs = tf.tile([[3], [0]], [1, max_length - 2])
        expected_outputs = tf.concat([inputs, expected_outputs], axis=1)
        self.assertAllEqual(outputs, expected_outputs)

    def test_from_logits(self):
        def token_logits_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.constant([[1.0, 2.0, 3.0, 4.0]])
            return tf.repeat(prob, batch_size, axis=0)

        def token_probability_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.keras.activations.softmax(
                tf.constant([[1.0, 2.0, 3.0, 4.0]])
            )
            return tf.repeat(prob, batch_size, axis=0)

        max_length = 5
        inputs = tf.constant([[0, 1], [1, 2]])
        tf.random.set_seed(42)
        output_logit = top_p_search(
            token_logits_fn,
            inputs,
            max_length=max_length,
            p=0.92,
            from_logits=True,
            seed=42,
        )
        tf.random.set_seed(42)
        output_probs = top_p_search(
            token_probability_fn,
            inputs,
            max_length=max_length,
            p=0.92,
            from_logits=False,
            seed=42,
        )
        self.assertAllEqual(output_logit, output_probs)
