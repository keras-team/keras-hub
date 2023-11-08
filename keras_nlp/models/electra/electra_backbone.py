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

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.backend import keras
from keras_nlp.layers.modeling.position_embedding import PositionEmbedding
from keras_nlp.layers.modeling.reversible_embedding import ReversibleEmbedding
from keras_nlp.layers.modeling.transformer_encoder import TransformerEncoder
from keras_nlp.models.backbone import Backbone
from keras_nlp.utils.keras_utils import gelu_approximate


def electra_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


@keras_nlp_export("keras_nlp.models.ElectraBackbone")
class ElectraBackbone(Backbone):
    """A Electra encoder network.

    This network implements a bidirectional Transformer-based encoder as
    described in ["Electra: Pre-training Text Encoders as Discriminators Rather
    Than Generators"](https://arxiv.org/abs/2003.10555). It includes the
    embedding lookups and transformer layers, but not the masked language model
    or classification task networks.

    The default constructor gives a fully customizable, randomly initialized
    Electra encoder with any number of layers, heads, and embedding
    dimensions.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://huggingface.co/docs/transformers/model_doc/electra#overview).

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        hidden_dim: int. The size of the transformer encoding and pooler layers.
        embedding_dim: int. The size of the token embeddings.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        dropout: float. Dropout probability for the Transformer encoder.
        max_sequence_length: int. The maximum sequence length that this encoder
            can consume. If None, `max_sequence_length` uses the value from
            sequence length. This determines the variable shape for positional
            embeddings.

    Examples:
        ```python
        input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "segment_ids": np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]]),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
        }
        # Randomly initialized Electra encoder
        backbone = keras_nlp.models.ElectraBackbone(
            vocabulary_size=1000,
            num_layers=2,
            num_heads=2,
            hidden_dim=32,
            intermediate_dim=64,
            dropout=0.1,
            max_sequence_length=512,
            )
        # Returns sequence and pooled outputs.
        sequence_output, pooled_output = backbone(input_data)
        ```
    """

    def __init__(
        self,
        vocab_size,
        num_layers,
        num_heads,
        hidden_dim,
        embedding_dim,
        intermediate_dim,
        dropout=0.1,
        max_sequence_length=512,
        num_segments=2,
        **kwargs,
    ):
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
        token_embedding_layer = ReversibleEmbedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            embeddings_initializer=electra_kernel_initializer(),
            name="token_embedding",
        )
        token_embedding = token_embedding_layer(token_id_input)
        position_embedding = PositionEmbedding(
            initializer=electra_kernel_initializer(),
            sequence_length=max_sequence_length,
            name="position_embedding",
        )(token_embedding)
        segment_embedding = keras.layers.Embedding(
            input_dim=num_segments,
            output_dim=embedding_dim,
            embeddings_initializer=electra_kernel_initializer(),
            name="segment_embedding",
        )(segment_id_input)

        # Add all embeddings together.
        x = keras.layers.Add()(
            (token_embedding, position_embedding, segment_embedding),
        )
        # Layer normalization
        x = keras.layers.LayerNormalization(
            name="embeddings_layer_norm",
            axis=-1,
            epsilon=1e-12,
            dtype="float32",
        )(x)
        # Dropout
        x = keras.layers.Dropout(
            dropout,
            name="embeddings_dropout",
        )(x)
        if hidden_dim != embedding_dim:
            x = keras.layers.Dense(
                hidden_dim,
                kernel_initializer=electra_kernel_initializer(),
                name="embeddings_projection",
            )(x)

        # Apply successive transformer encoder blocks.
        for i in range(num_layers):
            x = TransformerEncoder(
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                activation=gelu_approximate,
                dropout=dropout,
                layer_norm_epsilon=1e-12,
                kernel_initializer=electra_kernel_initializer(),
                name=f"transformer_layer_{i}",
            )(x, padding_mask=padding_mask)

        sequence_output = x
        x = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=electra_kernel_initializer(),
            activation="tanh",
            name="pooled_dense",
        )(x)
        pooled_output = x[:, cls_token_index, :]

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

        # All references to self below this line
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length
        self.num_segments = num_segments
        self.cls_token_index = cls_token_index
        self.token_embedding = token_embedding_layer

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "embedding_dim": self.embedding_dim,
                "intermediate_dim": self.intermediate_dim,
                "dropout": self.dropout,
                "max_sequence_length": self.max_sequence_length,
                "num_segments": self.num_segments,
            }
        )
        return config
