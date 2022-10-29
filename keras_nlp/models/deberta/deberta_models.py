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

"""DeBERTa backbone models."""

import tensorflow as tf
from tensorflow import keras

from keras_nlp.models.deberta.deberta_encoder import DebertaEncoder


def deberta_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


class DebertaCustomEncoder(keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        dropout=0.1,
        max_sequence_length=512,
        bucket_size=256,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length
        self.bucket_size = bucket_size

        self.rel_embeddings = self.add_weight(
            shape=(bucket_size * 2, hidden_dim),
            initializer=deberta_kernel_initializer(),
            name="rel_embedding",
        )

        self._rel_embedding_layer_norm = keras.layers.LayerNormalization(
            epsilon=1e-7
        )

        self.encoder_layers = [
            DebertaEncoder(
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                max_position_embeddings=max_sequence_length,
                bucket_size=bucket_size,
                dropout=dropout,
                activation=lambda x: keras.activations.gelu(
                    x,
                    approximate=True,
                ),
                kernel_initializer=deberta_kernel_initializer(),
                name=f"deberta_encoder_layer_{i}",
            )
            for i in range(num_layers)
        ]

    def call(self, inputs, padding_mask=None):
        x = inputs

        rel_embeddings = self._rel_embedding_layer_norm(self.rel_embeddings)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(
                inputs=x,
                rel_embeddings=rel_embeddings,
                padding_mask=padding_mask,
            )

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "dropout": self.dropout,
                "max_sequence_length": self.max_sequence_length,
            }
        )
        return config


class DebertaCustom(keras.Model):
    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        dropout=0.1,
        max_sequence_length=512,
        bucket_size=256,
        name=None,
        trainable=True,
    ):

        # Inputs
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        # Embed tokens.
        x = keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=deberta_kernel_initializer(),
            name="token_embedding",
        )(token_id_input)

        # Normalize and apply dropout to embeddings.
        x = keras.layers.LayerNormalization(
            name="embeddings_layer_norm",
            axis=-1,
            epsilon=1e-7,
            dtype=tf.float32,
        )(x)
        x = keras.layers.Dropout(
            dropout,
            name="embeddings_dropout",
        )(x)

        # Apply successive transformer encoder blocks.
        x = DebertaCustomEncoder(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            dropout=dropout,
            max_sequence_length=max_sequence_length,
            bucket_size=bucket_size,
            name="deberta_custom_encoder",
        )(x, padding_mask=padding_mask)

        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs={
                "token_ids": token_id_input,
                "padding_mask": padding_mask,
            },
            outputs=x,
            name=name,
            trainable=trainable,
        )
        # All references to `self` below this line
        self.vocabulary_size = vocabulary_size
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_sequence_length = max_sequence_length
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "max_sequence_length": self.max_sequence_length,
                "dropout": self.dropout,
            }
        )
        return config
