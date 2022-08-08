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
"""Bert model and layer implementations.

These components come from the tensorflow official model repository for BERT:
https://github.com/tensorflow/models/tree/master/official/nlp/modeling

This is to get us into a testable state. We should work to replace all of these
components with components from the keras-nlp library.
"""

import tensorflow as tf
from tensorflow import keras

import keras_nlp


class BertModel(keras.Model):
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
        dropout: Dropout probability for the Transformer encoder.
        num_attention_heads: The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        inner_size: The output dimension of the first Dense layer in a two-layer
            feedforward network for each transformer.
        inner_activation: The activation for the first Dense layer in a
            two-layer feedforward network for each transformer.
        initializer_range: The initialzer range to use for a truncated normal
            initializer.
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
        num_layers=12,
        hidden_size=768,
        dropout=0.1,
        num_attention_heads=12,
        inner_size=3072,
        inner_activation="gelu",
        initializer_range=0.02,
        max_sequence_length=512,
        type_vocab_size=2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.max_sequence_length = max_sequence_length
        self.type_vocab_size = type_vocab_size
        self.inner_size = inner_size
        self.inner_activation = keras.activations.get(inner_activation)
        self.initializer_range = initializer_range
        self.initializer = keras.initializers.TruncatedNormal(
            stddev=initializer_range
        )
        self.dropout = dropout

        self._embedding_layer = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=hidden_size,
            embeddings_initializer=self.initializer,
            name="word_embeddings",
        )

        self._position_embedding_layer = keras_nlp.layers.PositionEmbedding(
            initializer=self.initializer,
            sequence_length=max_sequence_length,
            name="position_embedding",
        )

        self._type_embedding_layer = keras.layers.Embedding(
            input_dim=type_vocab_size,
            output_dim=hidden_size,
            embeddings_initializer=self.initializer,
            name="type_embeddings",
        )

        self._embedding_norm_layer = keras.layers.LayerNormalization(
            name="embeddings/layer_norm",
            axis=-1,
            epsilon=1e-12,
            dtype=tf.float32,
        )

        self._embedding_dropout = keras.layers.Dropout(
            rate=dropout, name="embedding_dropout"
        )

        self._transformer_layers = []
        for i in range(num_layers):
            layer = keras_nlp.layers.TransformerEncoder(
                num_heads=num_attention_heads,
                intermediate_dim=inner_size,
                activation=self.inner_activation,
                dropout=dropout,
                kernel_initializer=self.initializer,
                name="transformer/layer_%d" % i,
            )
            self._transformer_layers.append(layer)

        # This is used as the intermediate output for the NSP prediction head.
        # It is important we include this in the mode, as we want to preserve
        # these weights for fine-tuning tasks.
        self._pooler_layer = keras.layers.Dense(
            units=hidden_size,
            activation="tanh",
            kernel_initializer=self.initializer,
            name="pooler_dense",
        )

        self.inputs = dict(
            input_ids=keras.Input(shape=(None,), dtype=tf.int32),
            input_mask=keras.Input(shape=(None,), dtype=tf.int32),
            segment_ids=keras.Input(shape=(None,), dtype=tf.int32),
        )

    def call(self, inputs):
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            input_mask = inputs.get("input_mask")
            segment_ids = inputs.get("segment_ids")
        else:
            raise ValueError(f"Inputs should be a dict. Received: {inputs}.")

        word_embeddings = None
        word_embeddings = self._embedding_layer(input_ids)
        position_embeddings = self._position_embedding_layer(word_embeddings)
        type_embeddings = self._type_embedding_layer(segment_ids)

        embeddings = word_embeddings + position_embeddings + type_embeddings
        embeddings = self._embedding_norm_layer(embeddings)
        embeddings = self._embedding_dropout(embeddings)

        x = embeddings
        for layer in self._transformer_layers:
            x = layer(x, padding_mask=input_mask)
        sequence_output = x
        pooled_output = self._pooler_layer(x[:, 0, :])  # 0 is the [CLS] token.
        return sequence_output, pooled_output

    def get_embedding_table(self):
        return self._embedding_layer.embeddings

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
