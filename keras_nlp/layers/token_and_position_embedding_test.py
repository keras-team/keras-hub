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

import tensorflow as tf
from tensorflow import keras

from keras_nlp.layers import TokenAndPositionEmbedding


class TokenAndPositionEmbeddingTest(tf.test.TestCase):
    def test_get_config_and_from_config(self):
        token_and_position_embed = TokenAndPositionEmbedding(
            vocabulary_size=5,
            max_length=10,
            embedding_dim=32,
        )

        config = token_and_position_embed.get_config()

        expected_config_subset = {
            "embeddings_initializer": keras.initializers.serialize(
                keras.initializers.GlorotUniform()
            ),
            "embeddings_regularizer": None,
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
