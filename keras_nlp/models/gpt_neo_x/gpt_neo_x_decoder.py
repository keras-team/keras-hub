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
import tensorflow as tf
from tensorflow import keras

from keras_nlp.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_nlp.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)
from keras_nlp.models.gpt_neo_x.gpt_neo_x_attention import GPTNeoXAttention
from keras_nlp.utils.keras_utils import clone_initializer


class GPTNeoXDecoder(keras.layers.Layer):
    """GPTNeoX decoder.

    This class follows the architecture of the GPT-NeoX decoder layer in the
    paper [GPT-NeoX-20B: An Open-Source Autoregressive Language Model](https://arxiv.org/abs/2204.06745).
    Users can instantiate multiple instances of this class to stack up a decoder.

    This layer will always apply a causal mask to the decoder attention layer.

    Args:
        intermediate_dim: int, the hidden size of feedforward network.
        num_heads: int, the number of heads for multi-head attention.
        dropout: float. the dropout value, shared by
            the multi-head attention and feedforward layers.
        activation: string or `keras.activations`. the activation function of
            feedforward network.
        layer_norm_epsilon: float. The epsilon value in layer
            normalization components.
        kernel_initializer: string or `keras.initializers` initializer. The
            kernel initializer for the dense and multi-head  attention layers.
        bias_initializer: string or `keras.initializers` initializer. The bias
            initializer for the dense and multi-head  attention layers.
        rotary_max_wavelength: int. The maximum angular wavelength of the
            sine/cosine curves, for rotary embeddings.
        rotary_percentage: float. The percentage by which query, key, value
            matrices are to be rotated.
        max_sequence_length: int. The maximum sequence length that this encoder
             can consume. If `None`, `max_sequence_length` uses the value from
             sequence length. This determines the variable shape for positional
             embeddings.
        name: string. The name of the layer.
    """

    def __init__(
        self,
        intermediate_dim,
        num_heads,
        dropout=0.0,
        activation="relu",
        layer_norm_epsilon=1e-5,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        rotary_percentage=0.25,
        rotary_max_wavelength=10000,
        max_sequence_length=512,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.rotary_percentage = rotary_percentage
        self.rotary_max_wavelength = rotary_max_wavelength
        self.max_sequence_length = max_sequence_length
        self.activation = keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.supports_masking = True
        self.rotary_percentage = rotary_percentage

    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        # Self attention layers.
        self._self_attention_layer = GPTNeoXAttention(
            num_heads=self.num_heads,
            hidden_dim=hidden_dim,
            dropout=self.dropout,
            rotary_percentage=self.rotary_percentage,
            rotary_max_wavelength=self.rotary_max_wavelength,
            max_sequence_length=self.max_sequence_length,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )

        self._self_attention_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )

        self._self_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
        )

        # Feedforward layers.
        self._feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        self._feedforward_output_dense = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )

        self._feedforward_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )

        self._feedforward_dropout = keras.layers.Dropout(
            rate=self.dropout,
        )

    def call(
        self,
        decoder_sequence,
        decoder_padding_mask=None,
        decoder_attention_mask=None,
    ):
        # Compute self attention mask.
        batch_size = tf.shape(decoder_sequence)[0]
        input_length = output_length = tf.shape(decoder_sequence)[1]

        self_attention_mask = compute_causal_mask(
            batch_size, input_length, output_length, 0
        )
        decoder_mask = merge_padding_and_attention_mask(
            decoder_sequence, decoder_padding_mask, decoder_attention_mask
        )
        if decoder_mask is not None:
            self_attention_mask = tf.minimum(decoder_mask, self_attention_mask)

        residual = decoder_sequence

        x = self._self_attention_layernorm(decoder_sequence)

        # Self attention block.
        x = self._self_attention_layer(
            hidden_states=x,
            attention_mask=self_attention_mask,
        )
        x = self._self_attention_dropout(x)
        attention_output = x

        x = self._feedforward_layernorm(decoder_sequence)
        x = self._feedforward_intermediate_dense(x)
        x = self._feedforward_output_dense(x)
        feedforward_output = x

        return feedforward_output + attention_output + residual

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "rotary_percentage": self.rotary_percentage,
                "rotary_max_wavelength": self.rotary_max_wavelength,
                "max_sequence_length": self.max_sequence_length,
                "activation": keras.activations.serialize(self.activation),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
            }
        )
        return config
