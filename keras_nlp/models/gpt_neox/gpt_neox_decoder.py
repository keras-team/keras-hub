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

from keras_nlp.models.gpt_neox.gpt_neox_attention import GPTNeoXAttention
from keras_nlp.utils.keras_utils import clone_initializer

from keras_nlp.layers.transformer_layer_utils import (  # isort:skip
    compute_causal_mask,
    merge_padding_and_attention_mask,
)


class GPTNeoXDecoder(keras.layers.Layer):
    def __init__(
        self,
        intermediate_dim,
        num_heads,
        max_position_embeddings=512,
        dropout=0.2,
        activation="relu",
        layer_norm_epsilon=1e-5,
        rotary_pct=0.25,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        normalize_first=None,
        name=None,
        **kwargs,
    ):
        self._input_shape = kwargs.pop("build_input_shape", None)
        self._has_cross_attention = kwargs.pop("has_cross_attention", False)

        super().__init__(name=name, **kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.rotary_pct = rotary_pct
        self.max_position_embeddings = max_position_embeddings
        self.activation = keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.normalize_first = normalize_first
        self._built = False
        self.supports_masking = True

        if self._input_shape is not None:
            self._build(self._input_shape)

    def _build(self, input_shape):
        # Create layers based on input shape.
        self._built = True
        self._input_shape = input_shape
        # Infer the dimension of our hidden feature size from the build shape.
        hidden_dim = input_shape[-1]

        # Self attention layers.
        self._self_attention_layer = GPTNeoXAttention(
            num_heads=self.num_heads,
            hidden_dim=hidden_dim,
            max_position_embeddings=self.max_position_embeddings,
            dropout=self.dropout,
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
        encoder_sequence=None,
        decoder_padding_mask=None,
        decoder_attention_mask=None,
        encoder_padding_mask=None,
        encoder_attention_mask=None,
    ):

        if not self._built:
            self._build(decoder_sequence.shape)

        #         has_self_attention_cache = self_attention_cache is not None

        x = decoder_sequence  # Intermediate result.

        # Compute self attention mask.
        batch_size = tf.shape(decoder_sequence)[0]
        input_length = output_length = tf.shape(decoder_sequence)[1]
        # We need to handle a rectangular causal mask when doing cached
        # decoding. For generative inference, `decoder_sequence` will
        # generally be length 1, and `cache` will be the full generation length.
        #         if self_attention_cache is not None:
        #             input_length = tf.shape(self_attention_cache)[2]
        self_attention_mask = compute_causal_mask(
            batch_size,
            input_length,
            output_length,
            #             0
            #             if self_attention_cache_update_index is None
            #             else self_attention_cache_update_index,
        )
        decoder_mask = merge_padding_and_attention_mask(
            decoder_sequence, decoder_padding_mask, decoder_attention_mask
        )
        if decoder_mask is not None:
            self_attention_mask = tf.minimum(decoder_mask, self_attention_mask)

        # Self attention block.
        residual = x
        if self.normalize_first:
            x = self._self_attention_layernorm(x)

        x = self._self_attention_layer(
            hidden_states=x,
            attention_mask=self_attention_mask,
            return_attention_scores=False,
        )
        x = self._self_attention_dropout(x)
        x = x + residual
        if not self.normalize_first:
            x = self._self_attention_layernorm(x)

        # Feedforward block.
        residual = x
        if self.normalize_first:
            x = self._feedforward_layernorm(x)

        x = self._feedforward_intermediate_dense(x)
        x = self._feedforward_output_dense(x)
        x = self._feedforward_dropout(x)
        # x = x + residual
        if not self.normalize_first:
            x = self._feedforward_layernorm(x)

        #         if self_attention_cache is not None:
        #             return (x, self_attention_cache)
        #         else:

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "activation": keras.activations.serialize(self.activation),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
                "normalize_first": self.normalize_first,
                "build_input_shape": self._input_shape,
                "has_cross_attention": self._has_cross_attention,
            }
        )
        return config
