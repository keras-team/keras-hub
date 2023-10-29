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

import copy

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.backend import keras
from keras_nlp.layers.modeling.token_and_position_embedding import (
    PositionEmbedding, ReversibleEmbedding
)
from keras_nlp.layers.modeling.transformer_encoder import TransformerEncoder
from keras_nlp.models.backbone import Backbone
from keras_nlp.utils.python_utils import classproperty


def electra_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)

@keras_nlp_export("keras_nlp.models.ElectraBackbone")
class ElectraBackbone(Backbone):
    """A Electra encoder network.

    This network implements a bi-directional Transformer-based encoder as
    described in ["Electra: Pre-training Text Encoders as Discriminators Rather
    Than Generators"](https://arxiv.org/abs/2003.10555). It includes the
    embedding lookups and transformer layers, but not the masked language model
    or classification task networks.

    The default constructor gives a fully customizable, randomly initialized
    Electra encoder with any number of layers, heads, and embedding
    dimensions. To load preset architectures and weights, use the
    `from_preset()` constructor.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://huggingface.co/docs/transformers/model_doc/electra#overview).
    """

    def __init__(
         self,
         vocabulary_size,
         num_layers,
         num_heads,
         embedding_size,
         hidden_size,
         intermediate_dim,
         dropout=0.1,
         max_sequence_length=512,
         num_segments=2,
         **kwargs
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
            input_dim=vocabulary_size,
            output_dim=embedding_size,
            embeddings_initializer=electra_kernel_initializer(),
            name="token_embedding",
        )
        token_embedding = token_embedding_layer(token_id_input)
        position_embedding = PositionEmbedding(
            input_dim=max_sequence_length,
            output_dim=embedding_size,
            merge_mode="add",
            embeddings_initializer=electra_kernel_initializer(),
            name="position_embedding",
        )(token_embedding)
        segment_embedding = keras.layers.Embedding(
            input_dim=max_sequence_length,
            output_dim=embedding_size,
            embeddings_initializer=electra_kernel_initializer(),
            name="segment_embedding",
        )(segment_id_input)

        # Add all embeddings together.
        x = keras.layers.Add()(
            (token_embedding, position_embedding, segment_embedding)
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
        # Project to hidden dim
        if hidden_size != embedding_size:
            x = keras.layers.Dense(
                hidden_size,
                kernel_initializer=electra_kernel_initializer(),
                name="embedding_projection",
            )(x)

        # Apply successive transformer encoder blocks.
        for i in range(num_layers):
            x = TransformerEncoder(
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                activation="gelu",
                dropout=dropout,
                layer_norm_epsilon=1e-12,
                kernel_initializer=electra_kernel_initializer(),
                name=f"transformer_layer_{i}",
            )(x, padding_mask=padding_mask)

        sequence_output = x
        x = keras.layers.Dense(
            hidden_size,
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
        self.vocab_size = vocabulary_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
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
                "hidden_size": self.hidden_size,
                "embedding_size": self.embedding_size,
                "intermediate_dim": self.intermediate_dim,
                "dropout": self.dropout,
                "max_sequence_length": self.max_sequence_length,
                "num_segments": self.num_segments,
                "cls_token_index": self.cls_token_index,
                "token_embedding": self.token_embedding,
            }
        )
        return config










