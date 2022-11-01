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


class RelativeEmbedding(keras.layers.Layer):
    def __init__(
        self,
        hidden_dim,
        bucket_size,
        epsilon,
        kernel_initializer,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.hidden_dim = hidden_dim
        self.bucket_size = bucket_size
        self.epsilon = epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

        self.rel_embeddings = self.add_weight(
            shape=(self.bucket_size * 2, self.hidden_dim),
            initializer=self.kernel_initializer,
            name="rel_embedding",
        )
        self.layer_norm = keras.layers.LayerNormalization(
            epsilon=epsilon, name="rel_embeddings_layer_norm"
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.shape(inputs)[-2]
        tf.print(sequence_length)

        rel_embeddings = self.rel_embeddings[:sequence_length, :]
        rel_embeddings = self.rel_embeddings[tf.newaxis, :]
        rel_embeddings = self.layer_norm(rel_embeddings)
        rel_embeddings = tf.repeat(rel_embeddings, repeats=batch_size, axis=0)

        return rel_embeddings

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "bucket_size": self.bucket_size,
                "epsilon": self.epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
            }
        )
        return config


class Deberta(keras.Model):
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
            epsilon=1e-7,
            dtype=tf.float32,
            name="embeddings_layer_norm",
        )(x)
        x = keras.layers.Dropout(
            dropout,
            name="embeddings_dropout",
        )(x)

        # Relative embedding layer.
        rel_embeddings = RelativeEmbedding(
            hidden_dim=hidden_dim,
            bucket_size=bucket_size,
            epsilon=1e-7,
            kernel_initializer=deberta_kernel_initializer(),
            name="rel_embedding",
        )(x)

        # Apply successive DeBERTa encoder blocks.
        for i in range(num_layers):
            x = DebertaEncoder(
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                max_position_embeddings=max_sequence_length,
                bucket_size=bucket_size,
                dropout=dropout,
                activation=lambda x: keras.activations.gelu(
                    x, approximate=True
                ),
                kernel_initializer=deberta_kernel_initializer(),
                name=f"deberta_encoder_layer_{i}",
            )(
                x,
                rel_embeddings=rel_embeddings,
                padding_mask=padding_mask,
            )

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
