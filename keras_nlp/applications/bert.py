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

import keras_nlp.layers

# isort: off
# TODO(jbischof): decide what to export or whether we are using these decorators
# from tensorflow.python.util.tf_export import keras_export

CLS_INDEX = 0
TOKEN_EMBEDDING_LAYER_NAME = "token_embedding"


# TODO(jbischof): move to keras_nlp/models
class Bert(keras.Model):
    """Bi-directional Transformer-based encoder network.

    This network implements a bi-directional Transformer-based encoder as
    described in "BERT: Pre-training of Deep Bidirectional Transformers for
    Language Understanding" (https://arxiv.org/abs/1810.04805). It includes the
    embedding lookups and transformer layers, but not the masked language model
    or classification task networks.

    The default values for this object are taken from the BERT-Base
    implementation in "BERT: Pre-training of Deep Bidirectional Transformers for
    Language Understanding".

    Args:
        vocab_size: The size of the token vocabulary.
        num_layers: The number of transformer layers.
        hidden_size: The size of the transformer hidden layers.
        num_attention_heads: The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        inner_size: The output dimension of the first Dense layer in a two-layer
            feedforward network for each transformer.
        inner_activation: The activation for the first Dense layer in a
            two-layer feedforward network for each transformer.
        initializer_range: The initialzer range to use for a truncated normal
            initializer.
        dropout: Dropout probability for the Transformer encoder.
        max_sequence_length: The maximum sequence length that this encoder can
            consume. If None, max_sequence_length uses the value from sequence
            length. This determines the variable shape for positional
            embeddings.
        type_vocab_size: The number of types that the 'segment_ids' input can
            take.
        norm_first: Whether to normalize inputs to attention and intermediate
            dense layers. If set False, output of attention and intermediate
            dense layers is normalized.
    """

    def __init__(
        self,
        vocab_size,
        num_layers,
        hidden_size,
        num_attention_heads,
        inner_size,
        inner_activation="gelu",
        initializer_range=0.02,
        dropout=0.1,
        max_sequence_length=512,
        type_vocab_size=2,
        **kwargs
    ):

        # Create lambda functions from input params
        inner_activation_fn = keras.activations.get(inner_activation)
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
        position_embedding = keras_nlp.layers.PositionEmbedding(
            initializer=initializer_fn,
            sequence_length=max_sequence_length,
            name="position_embedding",
        )(token_embedding)
        segment_embedding = keras.layers.Embedding(
            input_dim=type_vocab_size,
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
            x = keras_nlp.layers.TransformerEncoder(
                num_heads=num_attention_heads,
                intermediate_dim=inner_size,
                activation=inner_activation_fn,
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
        super(Bert, self).__init__(
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
            **kwargs
        )
        # All references to `self` below this line
        self.inner_activation_fn = inner_activation_fn
        self.initializer_fn = initializer_fn
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.max_sequence_length = max_sequence_length
        self.type_vocab_size = type_vocab_size
        self.inner_size = inner_size
        self.inner_activation = keras.activations.get(inner_activation)
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
                "num_attention_heads": self.num_attention_heads,
                "max_sequence_length": self.max_sequence_length,
                "type_vocab_size": self.type_vocab_size,
                "inner_size": self.inner_size,
                "inner_activation": keras.activations.serialize(
                    self.inner_activation
                ),
                "dropout": self.dropout,
                "initializer_range": self.initializer_range,
            }
        )
        return config


class BertClassifier(keras.Model):
    """Classifier model with BertEncoder."""

    # TODO(jbischof): figure out initialization default
    def __init__(self, encoder, num_classes, initializer, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.num_classes = num_classes
        self._logit_layer = keras.layers.Dense(
            num_classes,
            kernel_initializer=initializer,
            name="logits",
        )

    def call(self, inputs):
        # Ignore the sequence output, use the pooled output.
        pooled_output = self.encoder(inputs)["pooled_output"]
        return self._logit_layer(pooled_output)


def BertBase(weights=None):
    """Factory for BertEncoder using "Base" architecture."""

    model = Bert(
        vocab_size=30522,
        num_layers=12,
        hidden_size=768,
        num_attention_heads=12,
        inner_size=3072,
        inner_activation="gelu",
        initializer_range=0.02,
        dropout=0.1,
    )

    # TODO(jbischof): add some documentation or magic to load our checkpoints
    # Note: This is pure Keras and also intended to work with user checkpoints
    if weights is not None:
        model.load_weights(weights)

    # TODO(jbischof): attach the tokenizer
    return model
