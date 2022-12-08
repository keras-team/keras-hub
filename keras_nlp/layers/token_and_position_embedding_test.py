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
"""Tests for TokenAndPositionEmbedding"""

import os

import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.layers import TokenAndPositionEmbedding


class TokenAndPositionEmbeddingTest(tf.test.TestCase, parameterized.TestCase):
    def test_get_config_and_from_config(self):
        token_and_position_embed = TokenAndPositionEmbedding(
            vocabulary_size=5,
            sequence_length=10,
            embedding_dim=32,
        )

        config = token_and_position_embed.get_config()

        expected_config_subset = {
            "embeddings_initializer": keras.initializers.serialize(
                keras.initializers.GlorotUniform()
            ),
            "mask_zero": False,
        }

        self.assertEqual(config, {**config, **expected_config_subset})

        restored_token_and_position_embed = (
            TokenAndPositionEmbedding.from_config(config)
        )

        self.assertEqual(
            restored_token_and_position_embed.get_config(),
            {**config, **expected_config_subset},
        )

    def test_ragged_tensor(self):
        vocabulary_size = 5
        sequence_length = 4
        embedding_dim = 3
        test_layer = TokenAndPositionEmbedding(
            vocabulary_size=vocabulary_size,
            sequence_length=sequence_length,
            embedding_dim=embedding_dim,
            embeddings_initializer=keras.initializers.Constant(1.0),
        )
        # Create a 2-dimensional ragged input
        # (the first dimension is implicit).
        input_tensor = keras.Input(
            shape=(sequence_length,), dtype=tf.float32, ragged=True
        )
        output_tensor = test_layer(input_tensor)
        model = keras.Model(input_tensor, output_tensor)

        input_data = tf.ragged.constant(
            [
                [1.0, 1.0],
                [],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            ragged_rank=1,
        )
        expected_output_data = tf.ragged.constant(
            [
                [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                [],
                [
                    [2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0],
                ],
                [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
            ],
            ragged_rank=1,
        )
        output_data = model.predict(input_data)
        self.assertAllClose(output_data, expected_output_data)

    def test_dense_tensor(self):
        vocabulary_size = 5
        sequence_length = 4
        embedding_dim = 3
        test_layer = TokenAndPositionEmbedding(
            vocabulary_size=vocabulary_size,
            sequence_length=sequence_length,
            embedding_dim=embedding_dim,
            embeddings_initializer=keras.initializers.Constant(1.0),
        )
        # Create a 2-dimensional input
        # (the first dimension is implicit).
        inputs = keras.Input(shape=(sequence_length,), dtype="int32")
        outputs = test_layer(inputs)
        model = keras.Model(inputs, outputs)

        input_data = tf.ones((2, sequence_length), dtype="int32")
        expected_output_data = tf.ones((2, sequence_length, embedding_dim)) * 2
        output_data = model.predict(input_data)
        self.assertAllClose(output_data, expected_output_data)

    def test_mask_propagation(self):
        test_layer = TokenAndPositionEmbedding(
            vocabulary_size=5,
            sequence_length=4,
            embedding_dim=3,
            mask_zero=True,
        )
        input_data = tf.constant([[1, 0], [1, 0]])
        mask = input_data != 0
        outputs = test_layer(input_data)
        self.assertAllEqual(outputs._keras_mask, mask)

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saved_model(self, save_format, filename):
        vocabulary_size = 5
        sequence_length = 4
        embedding_dim = 3
        test_layer = TokenAndPositionEmbedding(
            vocabulary_size=vocabulary_size,
            sequence_length=sequence_length,
            embedding_dim=embedding_dim,
        )
        inputs = keras.Input(shape=(sequence_length,))
        outputs = test_layer(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        data = tf.zeros(shape=[2, sequence_length])
        model(data)

        path = os.path.join(self.get_temp_dir(), filename)
        model.save(path, save_format=save_format)
        loaded_model = keras.models.load_model(path)

        model_output = model.predict(data)
        loaded_model_output = loaded_model.predict(data)
        self.assertAllClose(model_output, loaded_model_output)
