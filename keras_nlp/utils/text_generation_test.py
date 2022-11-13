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
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.utils.text_generation import beam_search
from keras_nlp.utils.text_generation import greedy_search
from keras_nlp.utils.text_generation import random_search
from keras_nlp.utils.text_generation import top_k_search
from keras_nlp.utils.text_generation import top_p_search


class GreedySearchTextGenerationTest(tf.test.TestCase, parameterized.TestCase):
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

        def token_probability_fn(inputs):
            return model(inputs)[:, -1, :]

        self.token_probability_fn = token_probability_fn

    def test_generate_with_1d_prompt(self):
        inputs = tf.constant([1])
        outputs = greedy_search(self.token_probability_fn, inputs, max_length=5)
        self.assertEqual(outputs.shape, [5])

    def test_generate_with_2d_prompt(self):
        inputs = tf.constant([[1], [1]])
        outputs = greedy_search(self.token_probability_fn, inputs, max_length=5)
        self.assertEqual(outputs.shape, [2, 5])

    def test_generate_with_list_prompt(self):
        inputs = [[1], [1]]
        outputs = greedy_search(self.token_probability_fn, inputs, max_length=5)
        self.assertEqual(outputs.shape, [2, 5])

    def test_generate_with_ragged_prompt(self):
        def token_probability_fn(inputs):
            # Assert that user function is passed only dense tensors.
            self.assertIsInstance(inputs, tf.Tensor)
            prob = tf.constant([[0.0, 0.0, 0.0, 1.0]])
            return tf.repeat(prob, 2, axis=0)

        inputs = tf.ragged.constant([[1], [2, 1, 2]])
        outputs = greedy_search(token_probability_fn, inputs, max_length=5)
        self.assertEqual(outputs.shape, [2, 5])

    def test_assert_generation_is_correct(self):
        def token_probability_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.constant([[0.0, 0.0, 0.0, 1.0]])
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
            prob = tf.constant([[0.0, 0.0, 0.0, 1.0]])
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

    @parameterized.named_parameters(
        ("dense_jit_compile_false", False, False),
        ("dense_jit_compile_true", True, False),
        ("ragged_jit_compile_false", False, True),
    )
    def test_model_compile(self, jit_compile, ragged):
        def token_probability_fn(inputs):
            prob = tf.constant([[0.0, 0.0, 0.0, 1.0]])
            return tf.repeat(prob, 2, axis=0)

        max_length = 5

        class TestModel(tf.keras.Model):
            def call(self, inputs):
                generated = greedy_search(
                    token_probability_fn,
                    inputs,
                    max_length=max_length,
                    end_token_id=2,
                    pad_token_id=0,
                )
                return generated

        if ragged:
            inputs = tf.ragged.constant([[0, 1], [1, 1, 2]])
            expected_outputs = tf.constant([[0, 1, 3, 3, 3], [1, 1, 2, 0, 0]])
        else:
            inputs = tf.constant([[0, 1], [1, 2]])
            expected_outputs = [[0, 1, 3, 3, 3], [1, 2, 0, 0, 0]]

        model = TestModel()
        model.compile(jit_compile=jit_compile)

        outputs = model.predict(inputs)
        self.assertAllEqual(outputs, expected_outputs)

    @parameterized.named_parameters(
        ("dense_jit_compile_false", False, False),
        ("dense_jit_compile_true", True, False),
        ("ragged_jit_compile_false", False, True),
    )
    def test_model_compile_batched_ds(self, jit_compile, ragged):
        def token_probability_fn(inputs):
            prob = tf.constant([[0.0, 0.0, 0.0, 1.0]])
            return tf.repeat(prob, 2, axis=0)

        max_length = 5

        class TestModel(tf.keras.Model):
            def call(self, inputs):
                generated = greedy_search(
                    token_probability_fn,
                    inputs,
                    max_length=max_length,
                    end_token_id=2,
                    pad_token_id=0,
                )
                return generated

        if ragged:
            inputs = tf.ragged.constant([[0, 1], [1, 1, 2]])
            expected_outputs = tf.constant([[0, 1, 3, 3, 3], [1, 1, 2, 0, 0]])
        else:
            inputs = tf.constant([[0, 1], [1, 2]])
            expected_outputs = [[0, 1, 3, 3, 3], [1, 2, 0, 0, 0]]

        ds = tf.data.Dataset.from_tensor_slices(inputs).batch(2)

        model = TestModel()
        model.compile(jit_compile=jit_compile)

        outputs = model.predict(ds)
        self.assertAllEqual(outputs, expected_outputs)


class BeamSearchTextGenerationTest(tf.test.TestCase, parameterized.TestCase):
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

        def token_probability_fn(inputs):
            return model(inputs)[:, -1, :]

        self.token_probability_fn = token_probability_fn

    def test_generate_with_1d_prompt(self):
        inputs = tf.constant([1])
        outputs = beam_search(
            self.token_probability_fn,
            inputs,
            max_length=5,
            num_beams=5,
        )
        self.assertEqual(outputs.shape, [5])

    def test_generate_with_max_length_prompt(self):
        inputs = tf.ones(shape=(5,))
        outputs = beam_search(
            self.token_probability_fn,
            inputs,
            max_length=5,
            num_beams=5,
        )
        self.assertEqual(outputs.shape, [5])

        inputs = tf.ones(shape=(6,))
        outputs = beam_search(
            self.token_probability_fn,
            inputs,
            max_length=5,
            num_beams=5,
        )
        self.assertEqual(outputs.shape, [6])

    def test_generate_with_2d_prompt(self):
        inputs = tf.constant([[1], [1]])
        outputs = beam_search(
            self.token_probability_fn,
            inputs,
            max_length=5,
            num_beams=5,
        )
        self.assertEqual(outputs.shape, [2, 5])

    def test_generate_with_list_prompt(self):
        inputs = [[1], [1]]
        outputs = beam_search(
            self.token_probability_fn,
            inputs,
            max_length=5,
            num_beams=5,
        )
        self.assertEqual(outputs.shape, [2, 5])

    def test_generate_with_ragged_prompt(self):
        def token_probability_fn(inputs):
            # Assert that user function is passed only dense tensors.
            self.assertIsInstance(inputs, tf.Tensor)
            repeat_times = tf.shape(inputs)[0]
            prob = tf.constant([[0.0, 0.0, 0.0, 1.0]])
            return tf.repeat(prob, repeat_times, axis=0)

        inputs = tf.ragged.constant([[1], [2, 1, 2]])
        outputs = beam_search(
            token_probability_fn, inputs, max_length=5, num_beams=2
        )
        self.assertEqual(outputs.shape, [2, 5])

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

    def test_assert_generation_is_correct(self):
        def token_probability_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.constant([[0.0, 0.0, 0.0, 1.0]])
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
            prob = tf.constant([[0.0, 0.0, 0.0, 1.0]])
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

    @parameterized.named_parameters(
        ("dense_jit_compile_false", False, False),
        ("dense_jit_compile_true", True, False),
        ("ragged_jit_compile_false", False, True),
    )
    def test_model_compile(self, jit_compile, ragged):
        def token_probability_fn(inputs):
            repeat_times = tf.shape(inputs)[0]
            prob = tf.constant([[0.0, 0.0, 0.0, 1.0]])
            return tf.repeat(prob, repeat_times, axis=0)

        max_length = 5

        class TestModel(tf.keras.Model):
            def call(self, inputs):
                generated = beam_search(
                    token_probability_fn,
                    inputs,
                    max_length=max_length,
                    num_beams=2,
                    end_token_id=2,
                    pad_token_id=0,
                )
                return generated

        if ragged:
            inputs = tf.ragged.constant([[0, 1], [1, 1, 2]])
            expected_outputs = tf.constant([[0, 1, 3, 3, 3], [1, 1, 2, 0, 0]])
        else:
            inputs = tf.constant([[0, 1], [1, 2]])
            expected_outputs = [[0, 1, 3, 3, 3], [1, 2, 0, 0, 0]]

        model = TestModel()
        model.compile(jit_compile=jit_compile)

        outputs = model.predict(inputs)
        self.assertAllEqual(outputs, expected_outputs)

    @parameterized.named_parameters(
        ("dense_jit_compile_false", False, False),
        ("dense_jit_compile_true", True, False),
        ("ragged_jit_compile_false", False, True),
    )
    def test_model_compile_batched_ds(self, jit_compile, ragged):
        def token_probability_fn(inputs):
            repeat_times = tf.shape(inputs)[0]
            prob = tf.constant([[0.0, 0.0, 0.0, 1.0]])
            return tf.repeat(prob, repeat_times, axis=0)

        max_length = 5

        class TestModel(tf.keras.Model):
            def call(self, inputs):
                generated = beam_search(
                    token_probability_fn,
                    inputs,
                    max_length=max_length,
                    num_beams=2,
                    end_token_id=2,
                    pad_token_id=0,
                )
                return generated

        if ragged:
            inputs = tf.ragged.constant([[0, 1], [1, 1, 2]])
            expected_outputs = tf.constant([[0, 1, 3, 3, 3], [1, 1, 2, 0, 0]])
        else:
            inputs = tf.constant([[0, 1], [1, 2]])
            expected_outputs = [[0, 1, 3, 3, 3], [1, 2, 0, 0, 0]]

        ds = tf.data.Dataset.from_tensor_slices(inputs).batch(2)

        model = TestModel()
        model.compile(jit_compile=jit_compile)

        outputs = model.predict(ds)
        self.assertAllEqual(outputs, expected_outputs)


class RandomSearchTextGenerationTest(tf.test.TestCase, parameterized.TestCase):
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

        def token_probability_fn(inputs):
            return model(inputs)[:, -1, :]

        self.token_probability_fn = token_probability_fn

    def test_generate_with_1d_prompt(self):
        inputs = tf.constant([1])
        outputs = random_search(self.token_probability_fn, inputs, max_length=5)
        self.assertEqual(outputs.shape, [5])

    def test_generate_with_2d_prompt(self):
        inputs = tf.constant([[1], [1]])
        outputs = random_search(self.token_probability_fn, inputs, max_length=5)
        self.assertEqual(outputs.shape, [2, 5])

    def test_generate_with_list_prompt(self):
        inputs = [[1], [1]]
        outputs = random_search(self.token_probability_fn, inputs, max_length=5)
        self.assertEqual(outputs.shape, [2, 5])

    def test_generate_with_ragged_prompt(self):
        def token_probability_fn(inputs):
            # Assert that user function is passed only dense tensors.
            self.assertIsInstance(inputs, tf.Tensor)
            prob = tf.constant([[0.0, 0.0, 0.0, 1.0]])
            return tf.repeat(prob, 2, axis=0)

        inputs = tf.ragged.constant([[1], [2, 1, 2]])
        outputs = random_search(token_probability_fn, inputs, max_length=5)
        self.assertEqual(outputs.shape, [2, 5])

    def test_assert_probability_distribution_generation_is_correct(self):
        def token_probability_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.constant([[0.0, 0.0, 0.0, 1.0]])
            return tf.repeat(prob, batch_size, axis=0)

        batch_size = 10
        inputs = 3 * tf.ones([batch_size, 1], dtype=tf.int32)
        max_length = 3

        outputs_count = np.array([0, 0, 0, 0])
        tf.random.set_seed(42)
        for i in range(64):
            outputs = random_search(
                token_probability_fn, inputs, max_length=max_length, seed=42
            )
            flatten_predictions = tf.reshape(outputs[:, 1:], [-1])
            for pred in flatten_predictions:
                outputs_count[pred] += 1
        self.assertAllClose(
            outputs_count / np.sum(outputs_count),
            [0.0, 0.0, 0.0, 1.0],
            rtol=0.2,
        )

    def test_end_token_id(self):
        def token_probability_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.constant([[0.0, 0.0, 0.0, 1.0]])
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

    @parameterized.named_parameters(
        ("dense_jit_compile_false", False, False),
        ("dense_jit_compile_true", True, False),
        ("ragged_jit_compile_false", False, True),
    )
    def test_model_compile(self, jit_compile, ragged):
        def token_probability_fn(inputs):
            prob = tf.constant([[0.0, 0.0, 0.0, 1.0]])
            return tf.repeat(prob, 2, axis=0)

        max_length = 5

        class TestModel(tf.keras.Model):
            def call(self, inputs):
                generated = random_search(
                    token_probability_fn,
                    inputs,
                    max_length=max_length,
                    seed=42,
                    end_token_id=2,
                    pad_token_id=0,
                )
                return generated

        if ragged:
            inputs = tf.ragged.constant([[0, 1], [1, 1, 2]])
            expected_outputs = tf.constant([[0, 1, 3, 3, 3], [1, 1, 2, 0, 0]])
        else:
            inputs = tf.constant([[0, 1], [1, 2]])
            expected_outputs = [[0, 1, 3, 3, 3], [1, 2, 0, 0, 0]]

        model = TestModel()
        model.compile(jit_compile=jit_compile)

        outputs = model(inputs)
        self.assertAllEqual(outputs, expected_outputs)

    @parameterized.named_parameters(
        ("dense_jit_compile_false", False, False),
        ("dense_jit_compile_true", True, False),
        ("ragged_jit_compile_false", False, True),
    )
    def test_model_compile_batched_ds(self, jit_compile, ragged):
        def token_probability_fn(inputs):
            prob = tf.constant([[0.0, 0.0, 0.0, 1.0]])
            return tf.repeat(prob, 2, axis=0)

        max_length = 5

        class TestModel(tf.keras.Model):
            def call(self, inputs):
                generated = random_search(
                    token_probability_fn,
                    inputs,
                    max_length=max_length,
                    seed=42,
                    end_token_id=2,
                    pad_token_id=0,
                )
                return generated

        if ragged:
            inputs = tf.ragged.constant([[0, 1], [1, 1, 2]])
            expected_outputs = tf.constant([[0, 1, 3, 3, 3], [1, 1, 2, 0, 0]])
        else:
            inputs = tf.constant([[0, 1], [1, 2]])
            expected_outputs = [[0, 1, 3, 3, 3], [1, 2, 0, 0, 0]]

        ds = tf.data.Dataset.from_tensor_slices(inputs).batch(2)

        model = TestModel()
        model.compile(jit_compile=jit_compile)

        outputs = model.predict(ds)
        self.assertAllEqual(outputs, expected_outputs)

    def test_from_logits(self):
        def token_logits_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.constant([[1.0, 2.0, 3, 0, 4.0]])
            return tf.repeat(prob, batch_size, axis=0)

        def token_probability_fn(inputs):
            batch_size = inputs.shape[0]
            prob = keras.activations.softmax(
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


class TopKSearchTextGenerationTest(tf.test.TestCase, parameterized.TestCase):
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

        def token_probability_fn(inputs):
            return model(inputs)[:, -1, :]

        self.token_probability_fn = token_probability_fn

    def test_generate_with_1d_prompt(self):
        inputs = tf.constant([1])
        outputs = top_k_search(
            self.token_probability_fn, inputs, max_length=5, k=2
        )
        self.assertEqual(outputs.shape, [5])

    def test_generate_with_k_too_big(self):
        inputs = tf.constant([1])
        outputs = top_k_search(
            self.token_probability_fn,
            inputs,
            max_length=5,
            k=self.vocab_size + 1,
        )
        self.assertEqual(outputs.shape, [5])

    def test_generate_with_2d_prompt(self):
        inputs = tf.constant([[1], [1]])
        outputs = top_k_search(
            self.token_probability_fn, inputs, max_length=5, k=2
        )
        self.assertEqual(outputs.shape, [2, 5])

    def test_generate_with_list_prompt(self):
        inputs = [[1], [1]]
        outputs = top_k_search(
            self.token_probability_fn, inputs, max_length=5, k=2
        )
        self.assertEqual(outputs.shape, [2, 5])

    def test_generate_with_ragged_prompt(self):
        def token_probability_fn(inputs):
            # Assert that user function is passed only dense tensors.
            self.assertIsInstance(inputs, tf.Tensor)
            prob = tf.constant([[0.0, 0.0, 0.0, 1.0]])
            return tf.repeat(prob, 2, axis=0)

        inputs = tf.ragged.constant([[1], [2, 1, 2]])
        outputs = top_k_search(token_probability_fn, inputs, max_length=5, k=2)
        self.assertEqual(outputs.shape, [2, 5])

    def test_assert_probability_distribution_generation_is_correct(self):
        def token_probability_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.constant([[0.0, 0.0, 0.0, 1.0]])
            return tf.repeat(prob, batch_size, axis=0)

        batch_size = 10
        inputs = 3 * tf.ones([batch_size, 1], dtype=tf.int32)
        max_length = 3

        outputs_count = np.array([0, 0, 0, 0])
        tf.random.set_seed(42)
        for _ in range(64):
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
            [0.0, 0.0, 0.0, 1.0],
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
            prob = tf.constant([[0.0, 0.0, 0.0, 1.0]])
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

    @parameterized.named_parameters(
        ("dense_jit_compile_false", False, False),
        ("dense_jit_compile_true", True, False),
        ("ragged_jit_compile_false", False, True),
    )
    def test_model_compile(self, jit_compile, ragged):
        def token_probability_fn(inputs):
            prob = tf.constant([[0.0, 0.0, 0.0, 1.0]])
            return tf.repeat(prob, 2, axis=0)

        max_length = 5

        class TestModel(tf.keras.Model):
            def call(self, inputs):
                generated = top_k_search(
                    token_probability_fn,
                    inputs,
                    max_length=max_length,
                    k=2,
                    seed=42,
                    end_token_id=2,
                    pad_token_id=0,
                )
                return generated

        if ragged:
            inputs = tf.ragged.constant([[0, 1], [1, 1, 2]])
            expected_outputs = tf.constant([[0, 1, 3, 3, 3], [1, 1, 2, 0, 0]])
        else:
            inputs = tf.constant([[0, 1], [1, 2]])
            expected_outputs = [[0, 1, 3, 3, 3], [1, 2, 0, 0, 0]]

        model = TestModel()
        model.compile(jit_compile=jit_compile)

        outputs = model.predict(inputs)
        self.assertAllEqual(outputs, expected_outputs)

    @parameterized.named_parameters(
        ("dense_jit_compile_false", False, False),
        ("dense_jit_compile_true", True, False),
        ("ragged_jit_compile_false", False, True),
    )
    def test_model_compile_batched_ds(self, jit_compile, ragged):
        def token_probability_fn(inputs):
            prob = tf.constant([[0.0, 0.0, 0.0, 1.0]])
            return tf.repeat(prob, 2, axis=0)

        max_length = 5

        class TestModel(tf.keras.Model):
            def call(self, inputs):
                generated = top_k_search(
                    token_probability_fn,
                    inputs,
                    max_length=max_length,
                    k=2,
                    seed=42,
                    end_token_id=2,
                    pad_token_id=0,
                )
                return generated

        if ragged:
            inputs = tf.ragged.constant([[0, 1], [1, 1, 2]])
            expected_outputs = tf.constant([[0, 1, 3, 3, 3], [1, 1, 2, 0, 0]])
        else:
            inputs = tf.constant([[0, 1], [1, 2]])
            expected_outputs = [[0, 1, 3, 3, 3], [1, 2, 0, 0, 0]]

        ds = tf.data.Dataset.from_tensor_slices(inputs).batch(2)

        model = TestModel()
        model.compile(jit_compile=jit_compile)

        outputs = model.predict(ds)
        self.assertAllEqual(outputs, expected_outputs)

    def test_from_logits(self):
        def token_logits_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.constant([[1.0, 2.0, 3.0, 4.0]])
            return tf.repeat(prob, batch_size, axis=0)

        def token_probability_fn(inputs):
            batch_size = inputs.shape[0]
            prob = keras.activations.softmax(
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


class TopPSearchTextGenerationTest(tf.test.TestCase, parameterized.TestCase):
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

        def token_probability_fn(inputs):
            return model(inputs)[:, -1, :]

        self.token_probability_fn = token_probability_fn

    def test_generate_with_1d_prompt(self):
        inputs = tf.constant([1])
        outputs = top_p_search(
            self.token_probability_fn, inputs, max_length=5, p=0.8
        )
        self.assertEqual(outputs.shape, [5])

    def test_generate_with_2d_prompt(self):
        inputs = tf.constant([[1], [1]])
        outputs = top_p_search(
            self.token_probability_fn, inputs, max_length=5, p=0.8
        )
        self.assertEqual(outputs.shape, [2, 5])

    def test_generate_with_list_prompt(self):
        inputs = [[1], [1]]
        outputs = top_p_search(
            self.token_probability_fn, inputs, max_length=5, p=0.8
        )
        self.assertEqual(outputs.shape, [2, 5])

    def test_generate_with_ragged_prompt(self):
        def token_probability_fn(inputs):
            # Assert that user function is passed only dense tensors.
            self.assertIsInstance(inputs, tf.Tensor)
            prob = tf.constant([[0.0, 0.0, 0.0, 1.0]])
            return tf.repeat(prob, 2, axis=0)

        inputs = tf.ragged.constant([[1], [2, 1, 2]])
        outputs = top_p_search(
            token_probability_fn, inputs, max_length=5, p=0.8
        )
        self.assertEqual(outputs.shape, [2, 5])

    def test_assert_probability_distribution_generation_is_correct(self):
        def token_probability_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.constant([[0.0, 0.0, 0.0, 1.0]])
            return tf.repeat(prob, batch_size, axis=0)

        batch_size = 10
        inputs = 3 * tf.ones([batch_size, 1], dtype=tf.int32)
        max_length = 3

        outputs_count = np.array([0, 0, 0, 0])
        tf.random.set_seed(42)
        for i in range(64):
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
            [0.0, 0.0, 0.0, 1.0],
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
            prob = tf.constant([[0.0, 0.0, 0.0, 1.0]])
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

    @parameterized.named_parameters(
        ("dense_jit_compile_false", False, False),
        ("dense_jit_compile_true", True, False),
        ("ragged_jit_compile_false", False, True),
    )
    def test_model_compile(self, jit_compile, ragged):
        def token_probability_fn(inputs):
            prob = tf.constant([[0.0, 0.0, 0.0, 1.0]])
            return tf.repeat(prob, 2, axis=0)

        max_length = 5

        class TestModel(tf.keras.Model):
            def call(self, inputs):
                generated = top_p_search(
                    token_probability_fn,
                    inputs,
                    max_length=max_length,
                    p=0.92,
                    seed=1,
                    end_token_id=2,
                    pad_token_id=0,
                )
                return generated

        if ragged:
            inputs = tf.ragged.constant([[0, 1], [1, 1, 2]])
            expected_outputs = tf.constant([[0, 1, 3, 3, 3], [1, 1, 2, 0, 0]])
        else:
            inputs = tf.constant([[0, 1], [1, 2]])
            expected_outputs = [[0, 1, 3, 3, 3], [1, 2, 0, 0, 0]]

        model = TestModel()
        model.compile(jit_compile=jit_compile)

        outputs = model.predict(inputs)
        self.assertAllEqual(outputs, expected_outputs)

    @parameterized.named_parameters(
        ("dense_jit_compile_false", False, False),
        ("dense_jit_compile_true", True, False),
        ("ragged_jit_compile_false", False, True),
    )
    def test_model_compile_batched_ds(self, jit_compile, ragged):
        def token_probability_fn(inputs):
            prob = tf.constant([[0.0, 0.0, 0.0, 1.0]])
            return tf.repeat(prob, 2, axis=0)

        max_length = 5

        class TestModel(tf.keras.Model):
            def call(self, inputs):
                generated = top_p_search(
                    token_probability_fn,
                    inputs,
                    max_length=max_length,
                    p=0.92,
                    seed=1,
                    end_token_id=2,
                    pad_token_id=0,
                )
                return generated

        if ragged:
            inputs = tf.ragged.constant([[0, 1], [1, 1, 2]])
            expected_outputs = tf.constant([[0, 1, 3, 3, 3], [1, 1, 2, 0, 0]])
        else:
            inputs = tf.constant([[0, 1], [1, 2]])
            expected_outputs = [[0, 1, 3, 3, 3], [1, 2, 0, 0, 0]]

        ds = tf.data.Dataset.from_tensor_slices(inputs).batch(2)

        model = TestModel()
        model.compile(jit_compile=jit_compile)

        outputs = model.predict(ds)
        self.assertAllEqual(outputs, expected_outputs)

    def test_from_logits(self):
        def token_logits_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.constant([[1.0, 2.0, 3.0, 4.0]])
            return tf.repeat(prob, batch_size, axis=0)

        def token_probability_fn(inputs):
            batch_size = inputs.shape[0]
            prob = keras.activations.softmax(
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
