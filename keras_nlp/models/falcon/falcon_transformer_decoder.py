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
import math

from keras_nlp.backend import keras
from keras_nlp.backend import ops
from keras_nlp.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_nlp.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)
from keras_nlp.models.falcon.falcon_attention import FalconAttention


class FalconTransformerDecoder(keras.layers.Layer):
    def __init__(
        self,
        num_attention_heads,
        intermediate_dim,
        layer_norm_epsilon=1e-5,
        attention_dropout=0,
        feedforward_dropout=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_attention_heads = num_attention_heads
        self.intermediate_dim = intermediate_dim
        self.layer_norm_epsilon = layer_norm_epsilon
        self.attention_dropout = attention_dropout
        self.feedforward_dropout = feedforward_dropout

    def build(self, decoder_sequence_shape):
        self.hidden_dim = decoder_sequence_shape[-1]
        self._input_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon
        )
        self._input_layernorm.build(decoder_sequence_shape)

        # Attention layers.
        self.key_dim = self.hidden_dim // self.num_attention_heads
        self._attention_layer = FalconAttention(
            num_heads=self.num_attention_heads,
            attention_dropout=self.attention_dropout,
        )
        self._attention_layer.build(
            decoder_sequence_shape,
        )

        self._attention_dropout = keras.layers.Dropout(
            rate=self.attention_dropout,
            name="attention_dropout",
        )

        self._post_attention_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon
        )
        self._post_attention_layernorm.build(decoder_sequence_shape)

        # Feedforward layers.
        # TODO: use_bias should be an argument to the transformer to support
        # other sizes of models, e.g. 7B, that don't use bias.
        self._dense_h_to_4h = keras.layers.Dense(
            self.intermediate_dim,
            activation=keras.activations.gelu,
            use_bias=True,
            name="dense_h_to_4h",
        )
        self._dense_h_to_4h.build(decoder_sequence_shape)

        self._dense_4h_to_h = keras.layers.Dense(
            self.hidden_dim,
            use_bias=True,
            name="dense_4h_to_h",
        )
        self._dense_4h_to_h.build(
            (
                decoder_sequence_shape[0],
                decoder_sequence_shape[1],
                self.intermediate_dim,
            )
        )

        self._feedforward_dropout = keras.layers.Dropout(
            rate=self.feedforward_dropout,
            name="feedforward_dropout",
        )

        self.built = True

    def call(
        self,
        inputs,
        decoder_padding_mask=None,
        decoder_attention_mask=None,
        training=None,
    ):
        attention_mask = self._compute_self_attention_mask(
            decoder_sequence=inputs,
            decoder_padding_mask=decoder_padding_mask,
            decoder_attention_mask=decoder_attention_mask,
        )

        residual = inputs

        x = self._input_layernorm(inputs)

        alibi = self._build_alibi_tensor(
            self.num_attention_heads, decoder_padding_mask
        )

        # Attention block.
        x = self._attention_layer(
            inputs=x,
            attention_mask=attention_mask,
            alibi=alibi,
        )

        x = self._attention_dropout(x, training=training)

        x = x + residual
        residual = x

        x = self._post_attention_layernorm(x)

        x = self._dense_h_to_4h(x)
        x = self._dense_4h_to_h(x)

        x = self._feedforward_dropout(x, training=training)

        decoder_output = x + residual

        return decoder_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_attention_heads": self.num_attention_heads,
                "intermediate_dim": self.intermediate_dim,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "attention_dropout": self.attention_dropout,
                "feedforward_dropout": self.feedforward_dropout,
            }
        )
        return config

    def compute_output_shape(self, decoder_sequence_shape):
        return decoder_sequence_shape

    def _compute_self_attention_mask(
        self,
        decoder_sequence,
        decoder_padding_mask,
        decoder_attention_mask,
        self_attention_cache=None,
        self_attention_cache_update_index=None,
    ):
        decoder_mask = merge_padding_and_attention_mask(
            decoder_sequence, decoder_padding_mask, decoder_attention_mask
        )
        batch_size = ops.shape(decoder_sequence)[0]
        input_length = output_length = ops.shape(decoder_sequence)[1]
        # We need to handle a rectangular causal mask when doing cached
        # decoding. For generative inference, `decoder_sequence` will
        # generally be length 1, and `cache` will be the full generation length.
        if self_attention_cache is not None:
            input_length = ops.shape(self_attention_cache)[2]

        causal_mask = compute_causal_mask(
            batch_size,
            input_length,
            output_length,
            (
                0
                if self_attention_cache_update_index is None
                else self_attention_cache_update_index
            ),
        )
        return (
            ops.minimum(decoder_mask, causal_mask)
            if decoder_mask is not None
            else causal_mask
        )

    def _build_alibi_tensor(self, num_heads, attention_mask):
        _, seq_length = attention_mask.shape
        slopes = ops.convert_to_tensor(
            self._get_slopes(num_heads),
            dtype=self.compute_dtype,
        )
        arange_tensor = (
            (ops.cumsum(attention_mask, axis=-1) - 1) * attention_mask
        )[:, None, :]
        alibi = slopes[..., None] * arange_tensor
        return ops.expand_dims(
            ops.reshape(alibi, (num_heads, 1, seq_length)), 0
        )

    def _get_slopes(self, num_heads):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(num_heads).is_integer():
            return get_slopes_power_of_2(num_heads)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + self._get_slopes(2 * closest_power_of_2)[0::2][
                    : num_heads - closest_power_of_2
                ]
            )
