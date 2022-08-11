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
"""Bert model and layer implementations."""

import tensorflow as tf
from tensorflow import keras

from keras_nlp.layers import PositionEmbedding
from keras_nlp.layers import TransformerEncoder

CLS_INDEX = 0
TOKEN_EMBEDDING_LAYER_NAME = "token_embedding"


class Bert(keras.Model):
    """Bi-directional Transformer-based encoder network.

    This network implements a bi-directional Transformer-based encoder as
    described in "BERT: Pre-training of Deep Bidirectional Transformers for
    Language Understanding" (https://arxiv.org/abs/1810.04805). It includes the
    embedding lookups and transformer layers, but not the masked language model
    or classification task networks.

    Args:
        vocab_size: The size of the token vocabulary.
        num_layers: The number of transformer layers.
        hidden_size: The size of the transformer hidden layers.
        num_heads: The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        intermediate_dim: The output dimension of the first Dense layer in a
            two-layer feedforward network for each transformer.
        intermediate_activiation: The activation for the first Dense layer in a
            two-layer feedforward network for each transformer.
        initializer_range: The initialzer range to use for a truncated normal
            initializer.
        dropout: Dropout probability for the Transformer encoder.
        max_sequence_length: The maximum sequence length that this encoder can
            consume. If None, max_sequence_length uses the value from sequence
            length. This determines the variable shape for positional
            embeddings.
        num_segments: The number of types that the 'segment_ids' input can
            take.
    """

    def __init__(
        self,
        vocab_size,
        num_layers,
        hidden_size,
        num_heads,
        intermediate_dim,
        intermediate_activiation="gelu",
        initializer_range=0.02,
        dropout=0.1,
        max_sequence_length=512,
        num_segments=2,
        **kwargs,
    ):

        # Create lambda functions from input params
        intermediate_activiation_fn = keras.activations.get(
            intermediate_activiation
        )
        initializer_fn = keras.initializers.TruncatedNormal(
            stddev=initializer_range
        )

        # Functional version of model
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="input_ids"
        )
        segment_id_input = keras.Input(
            shape=(None,), dtype="int32", name="segment_ids"
        )
        # TODO(jbischof): improve handling of masking following
        # https://www.tensorflow.org/guide/keras/masking_and_padding
        input_mask = keras.Input(
            shape=(None,), dtype="int32", name="input_mask"
        )

        # Embed tokens, positions, and segment ids.
        token_embedding = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=hidden_size,
            name=TOKEN_EMBEDDING_LAYER_NAME,
        )(token_id_input)
        position_embedding = PositionEmbedding(
            initializer=initializer_fn,
            sequence_length=max_sequence_length,
            name="position_embedding",
        )(token_embedding)
        segment_embedding = keras.layers.Embedding(
            input_dim=num_segments,
            output_dim=hidden_size,
            name="segment_embedding",
        )(segment_id_input)

        # Sum, normailze and apply dropout to embeddings.
        x = keras.layers.Add(
            name="embedding_sum",
        )((token_embedding, position_embedding, segment_embedding))
        x = keras.layers.LayerNormalization(
            name="embeddings/layer_norm",
            axis=-1,
            epsilon=1e-12,
            dtype=tf.float32,
        )(x)
        x = keras.layers.Dropout(
            dropout,
            name="embedding_dropout",
        )(x)

        # Apply successive transformer encoder blocks.
        for i in range(num_layers):
            x = TransformerEncoder(
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                activation=intermediate_activiation_fn,
                dropout=dropout,
                kernel_initializer=initializer_fn,
                name="transformer/layer_%d" % i,
            )(x, padding_mask=input_mask)

        # Construct the two BERT outputs, and apply a dense to the pooled output.
        sequence_output = x
        pooled_output = keras.layers.Dense(
            hidden_size,
            activation="tanh",
            name="pooled_dense",
        )(x[:, CLS_INDEX, :])

        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs={
                "input_ids": token_id_input,
                "segment_ids": segment_id_input,
                "input_mask": input_mask,
            },
            # TODO(jbischof): Consider list output
            outputs={
                "sequence_output": sequence_output,
                "pooled_output": pooled_output,
            },
            **kwargs,
        )
        # All references to `self` below this line
        self.intermediate_activiation_fn = intermediate_activiation_fn
        self.initializer_fn = initializer_fn
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_sequence_length = max_sequence_length
        self.num_segments = num_segments
        self.intermediate_dim = intermediate_dim
        self.intermediate_activiation = keras.activations.get(
            intermediate_activiation
        )
        self.initializer_range = initializer_range
        self.dropout = dropout

    def get_embedding_table(self):
        return self.get_layer(TOKEN_EMBEDDING_LAYER_NAME).embeddings

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "max_sequence_length": self.max_sequence_length,
                "num_segments": self.num_segments,
                "intermediate_dim": self.intermediate_dim,
                "intermediate_activiation": keras.activations.serialize(
                    self.intermediate_activiation
                ),
                "dropout": self.dropout,
                "initializer_range": self.initializer_range,
            }
        )
        return config


class BertClassifier(keras.Model):
    """
    Classifier model with BertEncoder.

    Args:
        encoder: A `Bert` Model to encode inputs.
        num_classes: Number of classes to predict.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
    """

    def __init__(
        self,
        encoder,
        num_classes,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.encoder = encoder
        self.num_classes = num_classes
        self._logit_layer = keras.layers.Dense(
            num_classes,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name="logits",
        )

    def call(self, inputs):
        # Ignore the sequence output, use the pooled output.
        pooled_output = self.encoder(inputs)["pooled_output"]
        return self._logit_layer(pooled_output)


def BertBase(**kwargs):
    """
    Factory for BertEncoder using "Base" architecture.

    This network implements a bi-directional Transformer-based encoder as
    described in "BERT: Pre-training of Deep Bidirectional Transformers for
    Language Understanding" (https://arxiv.org/abs/1810.04805). It includes the
    embedding lookups and transformer layers, but not the masked language model
    or classification task networks.
    """

    model = Bert(
        vocab_size=30522,
        num_layers=12,
        hidden_size=768,
        num_heads=12,
        intermediate_dim=3072,
        intermediate_activiation="gelu",
        initializer_range=0.02,
        dropout=0.1,
        **kwargs,
    )

    # TODO(jbischof): add some documentation or magic to load our checkpoints
    # TODO(jbischof): attach the tokenizer
    return model
