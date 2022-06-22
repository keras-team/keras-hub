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


def make_attention_mask(inputs, mask):
    """Make a 3D attention mask from a 2D input mask.

    Given `inputs` with shape `[batch, from_seq_length, ...]` and a mask with
    shape `[batch_size, to_seq_length]`, this will output a mask with dtype
    matching `inputs` with shape `[batch, from_seq_length, to_seq_length]`.

    Args:
        inputs: The inputs to the attention layer.
        mask: The 2D input mask.
    """
    inputs_shape = tf.shape(inputs)
    batch_size = inputs_shape[0]
    from_seq_length = inputs_shape[1]
    to_seq_length = tf.shape(mask)[1]
    mask = tf.reshape(mask, [batch_size, 1, to_seq_length])
    mask = tf.cast(mask, dtype=inputs.dtype)
    return tf.ones([batch_size, from_seq_length, 1], dtype=inputs.dtype) * mask


# TODO(mattdangerw): This class is needed for TPU friendly embeddings, we should
# remove it entirely and fix tf.keras.layers.Embedding as needed.
class OnDeviceEmbedding(keras.layers.Layer):
    """Performs an embedding lookup suitable for TPU devices.

    This layer uses either tf.gather or tf.one_hot to translate integer indices
    to float embeddings.

    Args:
        vocab_size: Number of elements in the vocabulary.
        embedding_width: Output size of the embedding layer.
        initializer: The initializer to use for the embedding weights. Defaults
            to "glorot_uniform".
        use_one_hot: Whether to use tf.one_hot over tf.gather for the embedding
            lookup. Defaults to False (that is, using tf.gather). Setting this
            option to True may improve performance, especially on small
            vocabulary sizes, but will generally require more memory.
      scale_factor: Whether to scale the output embeddings. Defaults to None
        (that is, not to scale). Setting this option to a float will let values
        in output embeddings multiplied by scale_factor.
    """

    def __init__(
        self,
        vocab_size,
        embedding_width,
        initializer="glorot_uniform",
        use_one_hot=False,
        scale_factor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._vocab_size = vocab_size
        self._embedding_width = embedding_width
        self._initializer = initializer
        self._use_one_hot = use_one_hot
        self._scale_factor = scale_factor

    def get_config(self):
        config = {
            "vocab_size": self._vocab_size,
            "embedding_width": self._embedding_width,
            "initializer": self._initializer,
            "use_one_hot": self._use_one_hot,
            "scale_factor": self._scale_factor,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            "embeddings",
            shape=[self._vocab_size, self._embedding_width],
            initializer=self._initializer,
            dtype=tf.float32,
        )

        super().build(input_shape)

    def call(self, inputs):
        flat_inputs = tf.reshape(inputs, [-1])
        if self._use_one_hot:
            one_hot_data = tf.one_hot(
                flat_inputs, depth=self._vocab_size, dtype=self.compute_dtype
            )
            embeddings = tf.matmul(one_hot_data, self.embeddings)
        else:
            embeddings = tf.gather(self.embeddings, flat_inputs)
        embeddings = tf.reshape(
            embeddings,
            tf.concat([tf.shape(inputs), [self._embedding_width]], axis=0),
        )
        embeddings.set_shape(inputs.shape.as_list() + [self._embedding_width])
        if self._scale_factor:
            embeddings *= self._scale_factor
        return embeddings


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

        self._embedding_layer = OnDeviceEmbedding(
            vocab_size=vocab_size,
            embedding_width=hidden_size,
            initializer=self.initializer,
            name="word_embeddings",
        )

        self._position_embedding_layer = keras_nlp.layers.PositionEmbedding(
            initializer=self.initializer,
            sequence_length=max_sequence_length,
            name="position_embedding",
        )

        self._type_embedding_layer = OnDeviceEmbedding(
            vocab_size=type_vocab_size,
            embedding_width=hidden_size,
            initializer=self.initializer,
            use_one_hot=True,
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
        return x

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
