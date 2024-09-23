# Copyright 2024 The KerasHub Authors
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

import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.position_embedding import PositionEmbedding
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.layers.modeling.transformer_encoder import TransformerEncoder
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.utils.keras_utils import gelu_approximate


def albert_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


@keras_hub_export("keras_hub.models.AlbertBackbone")
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
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights. Note that some computations,
            such as softmax and layer normalization, will always be done at
            float32 precision regardless of dtype.

    Example:
    ```python
    input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "segment_ids": np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]]),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
    }

    # Randomly initialized ALBERT encoder
    model = keras_hub.models.AlbertBackbone(
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
        dtype=None,
        **kwargs,
    ):
        if num_layers % num_groups != 0:
            raise ValueError(
                "`num_layers` must be divisible by `num_groups`. Received: "
                f"`num_layers={num_layers}` and `num_groups={num_groups}`."
            )

        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=embedding_dim,
            embeddings_initializer=albert_kernel_initializer(),
            dtype=dtype,
            name="token_embedding",
        )
        self.position_embedding = PositionEmbedding(
            initializer=albert_kernel_initializer(),
            sequence_length=max_sequence_length,
            dtype=dtype,
            name="position_embedding",
        )
        self.segment_embedding = keras.layers.Embedding(
            input_dim=num_segments,
            output_dim=embedding_dim,
            embeddings_initializer=albert_kernel_initializer(),
            dtype=dtype,
            name="segment_embedding",
        )
        self.embeddings_add = keras.layers.Add(
            dtype=dtype,
            name="embeddings_add",
        )
        self.embeddings_layer_norm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-12,
            dtype=dtype,
            name="embeddings_layer_norm",
        )
        self.embeddings_dropout = keras.layers.Dropout(
            dropout,
            dtype=dtype,
            name="embeddings_dropout",
        )
        self.embeddings_projection = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=albert_kernel_initializer(),
            dtype=dtype,
            name="embedding_projection",
        )
        self.transformer_layers = []
        for group_idx in range(num_groups):
            inner_layers = []
            for inner_idx in range(num_inner_repetitions):
                layer = TransformerEncoder(
                    num_heads=num_heads,
                    intermediate_dim=intermediate_dim,
                    activation=gelu_approximate,
                    dropout=dropout,
                    layer_norm_epsilon=1e-12,
                    kernel_initializer=albert_kernel_initializer(),
                    dtype=dtype,
                    name=f"group_{group_idx}_inner_layer_{inner_idx}",
                )
                inner_layers.append(layer)
            self.transformer_layers.append(inner_layers)
        self.pooled_dense = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=albert_kernel_initializer(),
            activation="tanh",
            dtype=dtype,
            name="pooled_dense",
        )

        # === Functional Model ===
        # Inputs
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        segment_id_input = keras.Input(
            shape=(None,), dtype="int32", name="segment_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        # Embed tokens, positions, and segment ids.
        tokens = self.token_embedding(token_id_input)
        positions = self.position_embedding(tokens)
        segments = self.segment_embedding(segment_id_input)
        # Sum, normalize and apply dropout to embeddings.
        x = self.embeddings_add((tokens, positions, segments))
        x = self.embeddings_layer_norm(x)
        x = self.embeddings_dropout(x)
        x = self.embeddings_projection(x)
        # Call transformer layers with repeated groups.
        num_calls_per_group = num_layers // num_groups
        for group in self.transformer_layers:
            for _ in range(num_calls_per_group):
                for transformer_layer in group:
                    x = transformer_layer(x, padding_mask=padding_mask_input)
        # Construct the two ALBERT outputs. The pooled output is a dense layer
        # on top of the [CLS] token.
        sequence_output = x
        cls_token_index = 0
        pooled_output = self.pooled_dense(x[:, cls_token_index, :])
        super().__init__(
            inputs={
                "token_ids": token_id_input,
                "segment_ids": segment_id_input,
                "padding_mask": padding_mask_input,
            },
            outputs={
                "sequence_output": sequence_output,
                "pooled_output": pooled_output,
            },
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
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
