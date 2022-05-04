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

from keras_nlp.utils.text_generation import generate_text_greedy


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

        def get_next_token_probability_fn(inputs):
            return model(inputs)[:, -1, :]

        self.get_next_token_probability_fn = get_next_token_probability_fn

    def test_generate_with_1d_input_ids(self):
        inputs = tf.constant([1])
        outputs = generate_text_greedy(
            self.get_next_token_probability_fn, inputs, max_length=5
        )
        self.assertEquals(outputs.shape, [5])

    def test_generate_with_2d_input_ids(self):
        inputs = tf.constant([[1], [1]])
        outputs = generate_text_greedy(
            self.get_next_token_probability_fn, inputs, max_length=5
        )
        self.assertEquals(outputs.shape, [2, 5])

    def test_empty_input_ids(self):
        inputs = tf.constant([])
        with self.assertRaisesRegex(ValueError, "input_ids must not be empty*"):
            generate_text_greedy(
                self.get_next_token_probability_fn, inputs, max_length=5
            )

    def test_assert_generation_is_correct(self):
        def get_next_token_probability_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.constant([[0.1, 0.2, 0.3, 0.4]])
            return tf.repeat(prob, batch_size, axis=0)

        batch_size = 10
        inputs = tf.ones([batch_size, 1], dtype=tf.int32)
        outputs = generate_text_greedy(
            get_next_token_probability_fn, inputs, max_length=3
        )
        self.assertAllEqual(outputs[:, -1], tf.repeat(3, batch_size))

    def test_end_token(self):
        def get_next_token_probability_fn(inputs):
            batch_size = inputs.shape[0]
            prob = tf.constant([[0.1, 0.2, 0.3, 0.4]])
            return tf.repeat(prob, batch_size, axis=0)

        max_length = 3
        inputs = tf.constant([[0], [3]])
        outputs = generate_text_greedy(
            get_next_token_probability_fn,
            inputs,
            max_length=max_length,
            end_token_id=0,
        )
        self.assertAllEqual(outputs[0, :], tf.repeat(0, max_length))
        self.assertAllEqual(outputs[1, :], tf.repeat(3, max_length))
