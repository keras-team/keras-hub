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

"""ALBERT backbone model."""

import copy

import tensorflow as tf
from tensorflow import keras

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.layers.position_embedding import PositionEmbedding
from keras_nlp.layers.transformer_encoder import TransformerEncoder
from keras_nlp.models.albert.albert_presets import backbone_presets
from keras_nlp.models.backbone import Backbone
from keras_nlp.utils.python_utils import classproperty


def albert_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


@keras_nlp_export("keras_nlp.models.AlbertBackbone")
class AlbertBackbone(Backbone):
    """ALBERT encoder network.

    This class implements a bi-directional Transformer-based encoder as
    described in
    ["ALBERT: A Lite BERT for Self-supervised Learning of Language Representations"](https://arxiv.org/abs/1909.11942).
    ALBERT is a more efficient variant of BERT, and uses parameter reduction
    techniques such as cross-layer parameter sharing and factorized embedding
    parameterization. This model class includes the embedding lookups and
    transformer layers, but not the masked language model or sentence order
    prediction heads.

    The default constructor gives a fully customizable, randomly initialized
    ALBERT encoder with any number of layers, heads, and embedding dimensions.
    To load preset architectures and weights, use the `from_preset` constructor.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int, must be divisible by `num_groups`. The number of
            "virtual" layers, i.e., the total number of times the input sequence
            will be fed through the groups in one forward pass. The input will
            be routed to the correct group based on the layer index.
        num_heads: int. The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        embedding_dim: int. The size of the embeddings.
        hidden_dim: int. The size of the transformer encoding and pooler layers.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        num_groups: int. Number of groups, with each group having
            `num_inner_repetitions` number of `TransformerEncoder` layers.
        num_inner_repetitions: int. Number of `TransformerEncoder` layers per
            group.
        dropout: float. Dropout probability for the Transformer encoder.
        max_sequence_length: int. The maximum sequence length that this encoder
            can consume. If None, `max_sequence_length` uses the value from
            sequence length. This determines the variable shape for positional
            embeddings.
        num_segments: int. The number of types that the 'segment_ids' input can
            take.

    Examples:
    ```python
    input_data = {
        "token_ids": tf.ones(shape=(1, 12), dtype=tf.int64),
        "segment_ids": tf.constant(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
        ),
        "padding_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
        ),
    }

    # Randomly initialized ALBERT encoder
    model = keras_nlp.models.AlbertBackbone(
        vocabulary_size=30000,
        num_layers=12,
        num_heads=12,
        num_groups=1,
        num_inner_repetitions=1,
        embedding_dim=128,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=12,
    )
    output = model(input_data)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_heads,
        embedding_dim,
        hidden_dim,
        intermediate_dim,
        num_groups=1,
        num_inner_repetitions=1,
        dropout=0.0,
        max_sequence_length=512,
        num_segments=2,
        **kwargs,
    ):
        if num_layers % num_groups != 0:
            raise ValueError(
                "`num_layers` must be divisible by `num_groups`. Received: "
                f"`num_layers={num_layers}` and `num_groups={num_groups}`."
            )

        # Index of classification token in the vocabulary
        cls_token_index = 0
        # Inputs
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        segment_id_input = keras.Input(
            shape=(None,), dtype="int32", name="segment_ids"
        )
        padding_mask = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        # Embed tokens, positions, and segment ids.
        token_embedding_layer = keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=embedding_dim,
            embeddings_initializer=albert_kernel_initializer(),
            name="token_embedding",
        )
        token_embedding = token_embedding_layer(token_id_input)
        position_embedding = PositionEmbedding(
            initializer=albert_kernel_initializer(),
            sequence_length=max_sequence_length,
            name="position_embedding",
        )(token_embedding)
        segment_embedding = keras.layers.Embedding(
            input_dim=num_segments,
            output_dim=embedding_dim,
            embeddings_initializer=albert_kernel_initializer(),
            name="segment_embedding",
        )(segment_id_input)

        # Sum, normalize and apply dropout to embeddings.
        x = keras.layers.Add()(
            (token_embedding, position_embedding, segment_embedding)
        )
        x = keras.layers.LayerNormalization(
            name="embeddings_layer_norm",
            axis=-1,
            epsilon=1e-12,
            dtype=tf.float32,
        )(x)
        x = keras.layers.Dropout(
            dropout,
            name="embeddings_dropout",
        )(x)

        # Project the embedding to `hidden_dim`.
        x = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=albert_kernel_initializer(),
            name="embedding_projection",
        )(x)

        def get_group_layer(group_idx):
            """Defines a group `num_inner_repetitions` transformer layers and
            returns the callable.
            """
            transformer_layers = [
                TransformerEncoder(
                    num_heads=num_heads,
                    intermediate_dim=intermediate_dim,
                    activation=lambda x: keras.activations.gelu(
                        x, approximate=True
                    ),
                    dropout=dropout,
                    layer_norm_epsilon=1e-12,
                    kernel_initializer=albert_kernel_initializer(),
                    name=f"group_{group_idx}_inner_layer_{inner_idx}",
                )
                for inner_idx in range(num_inner_repetitions)
            ]

            def call(x, padding_mask):
                for transformer_layer in transformer_layers:
                    x = transformer_layer(x, padding_mask=padding_mask)
                return x

            return call

        num_calls_per_group = num_layers // num_groups
        for group_idx in range(num_groups):
            # Define the group. A group in ALBERT terminology is any number of
            # repeated attention and FFN blocks.
            group_layer = get_group_layer(group_idx)

            # Assume num_layers = 8, num_groups = 4. Then, the order of group
            # calls will be 0, 0, 1, 1, 2, 2, 3, 3.
            for call in range(num_calls_per_group):
                x = group_layer(x, padding_mask=padding_mask)

        # Construct the two ALBERT outputs. The pooled output is a dense layer on
        # top of the [CLS] token.
        sequence_output = x
        pooled_output = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=albert_kernel_initializer(),
            activation="tanh",
            name="pooled_dense",
        )(x[:, cls_token_index, :])

        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs={
                "token_ids": token_id_input,
                "segment_ids": segment_id_input,
                "padding_mask": padding_mask,
            },
            outputs={
                "sequence_output": sequence_output,
                "pooled_output": pooled_output,
            },
            **kwargs,
        )
        # All references to `self` below this line
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.num_inner_repetitions = num_inner_repetitions
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length
        self.num_segments = num_segments
        self.cls_token_index = cls_token_index

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "num_groups": self.num_groups,
                "num_inner_repetitions": self.num_inner_repetitions,
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "dropout": self.dropout,
                "max_sequence_length": self.max_sequence_length,
                "num_segments": self.num_segments,
            }
        )
        return config

    @property
    def token_embedding(self):
        return self.get_layer("token_embedding")

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)
