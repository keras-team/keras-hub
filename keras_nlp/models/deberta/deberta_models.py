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
        layer_norm_epsilon=1e-05,
        kernel_initializer="glorot_uniform",
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.hidden_dim = hidden_dim
        self.bucket_size = bucket_size
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

        self.rel_embeddings = self.add_weight(
            shape=(self.bucket_size * 2, self.hidden_dim),
            initializer=self.kernel_initializer,
            name="rel_embedding",
        )
        self.layer_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon, name="rel_embeddings_layer_norm"
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        rel_embeddings = self.rel_embeddings[tf.newaxis, :]
        rel_embeddings = self.layer_norm(rel_embeddings)

        # Repeat `rel_embeddings` along axis = 0 for `batch_size` times. The
        # resultant shape is `(batch_size, bucket_size * 2, hidden_dim)`.
        rel_embeddings = tf.repeat(rel_embeddings, repeats=batch_size, axis=0)

        return rel_embeddings

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "bucket_size": self.bucket_size,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
            }
        )
        return config


class Deberta(keras.Model):
    """DeBERTa encoder network.

    This network implements a bi-directional Transformer-based encoder as
    described in
    ["DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing"](https://arxiv.org/abs/2111.09543).
    It includes the embedding lookups and transformer layers, but does not
    include the masked language modeling head used during pretraining.

    The default constructor gives a fully customizable, randomly initialized
    DeBERTa encoder with any number of layers, heads, and embedding
    dimensions. To load preset architectures and weights, use the `from_presets`
    constructor.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        hidden_dim: int. The size of the transformer encoding layer.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        dropout: float. Dropout probability for the Transformer encoder.
        max_sequence_length: int. The maximum sequence length this encoder can
            consume. The sequence length of the input must be less than
            `max_sequence_length`.
        bucket_size: int. The size of the relative position buckets. Generally
            kept as `max_sequence_length // 2`.

    Example usage:
    ```python
    input_data = {
        "token_ids": tf.random.uniform(
            shape=(1, 12), dtype=tf.int64, maxval=50265),
        "padding_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)),
    }

    # Randomly initialized RoBERTa model
    model = keras_nlp.models.Roberta(
        vocabulary_size=50265,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=12
    )

    # Call the model on the input data.
    output = model(input_data)
    ```
    """

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
        **kwargs,
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
            layer_norm_epsilon=1e-7,
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

        # Set default for `name` if none given
        if "name" not in kwargs:
            kwargs["name"] = "backbone"

        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs={
                "token_ids": token_id_input,
                "padding_mask": padding_mask,
            },
            outputs=x,
            **kwargs,
        )
        # All references to `self` below this line
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length
        self.bucket_size = bucket_size

    def get_config(self):
        return {
            "vocabulary_size": self.vocabulary_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "intermediate_dim": self.intermediate_dim,
            "dropout": self.dropout,
            "max_sequence_length": self.max_sequence_length,
            "bucket_size": self.bucket_size,
            "name": self.name,
            "trainable": self.trainable,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @classmethod
    def from_preset(
        cls,
        preset,
        load_weights=True,
        **kwargs,
    ):
        raise NotImplementedError
