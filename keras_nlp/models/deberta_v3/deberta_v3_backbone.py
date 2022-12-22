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

"""DeBERTa backbone model."""

import copy
import os

import tensorflow as tf
from tensorflow import keras

from keras_nlp.models.deberta_v3.deberta_v3_presets import backbone_presets
from keras_nlp.models.deberta_v3.disentangled_attention_encoder import (
    DisentangledAttentionEncoder,
)
from keras_nlp.models.deberta_v3.relative_embedding import RelativeEmbedding
from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.python_utils import format_docstring


def deberta_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


@keras.utils.register_keras_serializable(package="keras_nlp")
class DebertaV3Backbone(keras.Model):
    """DeBERTa encoder network.

    This network implements a bi-directional Transformer-based encoder as
    described in
    ["DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing"](https://arxiv.org/abs/2111.09543).
    It includes the embedding lookups and transformer layers, but does not
    include the enhanced masked decoding head used during pretraining.

    The default constructor gives a fully customizable, randomly initialized
    DeBERTa encoder with any number of layers, heads, and embedding
    dimensions. To load preset architectures and weights, use the `from_presets`
    constructor.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/microsoft/DeBERTa).

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        hidden_dim: int. The size of the transformer encoding layer.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        dropout: float. Dropout probability for the DeBERTa model.
        max_sequence_length: int. The maximum sequence length this encoder can
            consume. The sequence length of the input must be less than
            `max_sequence_length`.
        bucket_size: int. The size of the relative position buckets. Generally
            equal to `max_sequence_length // 2`.

    Example usage:
    ```python
    input_data = {
        "token_ids": tf.ones(shape=(1, 12), dtype=tf.int64),
        "padding_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)),
    }

    # Randomly initialized DeBERTa model
    model = keras_nlp.models.DebertaV3Backbone(
        vocabulary_size=128100,
        num_layers=12,
        num_heads=6,
        hidden_dim=384,
        intermediate_dim=1536,
        max_sequence_length=512,
        bucket_size=256,
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
            x = DisentangledAttentionEncoder(
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                max_position_embeddings=max_sequence_length,
                bucket_size=bucket_size,
                dropout=dropout,
                activation=lambda x: keras.activations.gelu(
                    x, approximate=False
                ),
                kernel_initializer=deberta_kernel_initializer(),
                name=f"disentangled_attention_encoder_layer_{i}",
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
        self.start_token_index = 0

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

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)

    @classmethod
    @format_docstring(names=", ".join(backbone_presets))
    def from_preset(
        cls,
        preset,
        load_weights=True,
        **kwargs,
    ):
        """Instantiate DeBERTa model from preset architecture and weights.

        Args:
            preset: string. Must be one of {{names}}.
            load_weights: Whether to load pre-trained weights into model.
                Defaults to `True`.

        Examples:
        ```python
        input_data = {
            "token_ids": tf.ones(shape=(1, 12), dtype=tf.int64),
            "padding_mask": tf.constant(
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
            ),
        }

        # Load architecture and weights from preset
        model = keras_nlp.models.DebertaV3Backbone.from_preset("deberta_base")
        output = model(input_data)

        # Load randomly initialized model from preset architecture
        model = keras_nlp.models.DebertaV3Backbone.from_preset(
            "deberta_base", load_weights=False
        )
        output = model(input_data)
        ```
        """
        if preset not in cls.presets:
            raise ValueError(
                "`preset` must be one of "
                f"""{", ".join(cls.presets)}. Received: {preset}."""
            )
        metadata = cls.presets[preset]
        config = metadata["config"]
        model = cls.from_config({**config, **kwargs})

        if not load_weights:
            return model

        weights = keras.utils.get_file(
            "model.h5",
            metadata["weights_url"],
            cache_subdir=os.path.join("models", preset),
            file_hash=metadata["weights_hash"],
        )

        model.load_weights(weights)
        return model
