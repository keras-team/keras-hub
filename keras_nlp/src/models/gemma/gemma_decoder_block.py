# Copyright 2024 The KerasNLP Authors
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
from keras import ops

from keras_nlp.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_nlp.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)
from keras_nlp.src.models.gemma.gemma_attention import CachedGemmaAttention
from keras_nlp.src.models.gemma.rms_normalization import RMSNormalization


class GemmaDecoderBlock(keras.layers.Layer):
    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        head_dim,
        num_query_heads,
        num_key_value_heads,
        layer_norm_epsilon=1e-6,
        dropout=0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.intermediate_dim = intermediate_dim
        self.hidden_dim = hidden_dim
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout

        self.pre_attention_norm = RMSNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="pre_attention_norm",
        )

        self.attention = CachedGemmaAttention(
            head_dim=head_dim,
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            dropout=dropout,
            dtype=self.dtype_policy,
            name="attention",
        )

        if self.dropout > 0:
            self.attention_dropout = keras.layers.Dropout(rate=dropout)
            self.feedforward_dropout = keras.layers.Dropout(rate=dropout)

        self.pre_ffw_norm = RMSNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="pre_ffw_norm",
        )

        self.gating_ffw = keras.layers.EinsumDense(
            equation="btd,df->btf",
            output_shape=(None, self.intermediate_dim // 2),
            dtype=self.dtype_policy,
            name="ffw_gating",
        )

        self.gating_ffw_2 = keras.layers.EinsumDense(
            equation="btd,df->btf",
            output_shape=(None, self.intermediate_dim // 2),
            dtype=self.dtype_policy,
            name="ffw_gating_2",
        )

        self.ffw_linear = keras.layers.EinsumDense(
            equation="btf,fd->btd",
            output_shape=(None, self.hidden_dim),
            dtype=self.dtype_policy,
            name="ffw_linear",
        )

    def build(self, input_shape):
        self.pre_attention_norm.build(input_shape)
        self.attention.build(input_shape)

        shape = input_shape
        self.pre_ffw_norm.build(shape)
        self.gating_ffw.build(shape)
        self.gating_ffw_2.build(shape)

        shape = self.gating_ffw.compute_output_shape(shape)
        self.ffw_linear.build(shape)
        self.built = True

    def compute_output_shape(self, input_shape):
        # Isometric
        return input_shape

    def _compute_attention_mask(
        self, x, padding_mask, cache, cache_update_index
    ):
        decoder_mask = merge_padding_and_attention_mask(
            inputs=x, padding_mask=padding_mask, attention_mask=None
        )
        batch_size = ops.shape(x)[0]
        input_length = output_length = ops.shape(x)[1]
        if cache is not None:
            input_length = ops.shape(cache)[2]

        causal_mask = compute_causal_mask(
            batch_size=batch_size,
            input_length=input_length,
            output_length=output_length,
            cache_index=cache_update_index,
        )

        return (
            ops.minimum(decoder_mask, causal_mask)
            if decoder_mask is not None
            else causal_mask
        )

    def call(
        self,
        x,
        padding_mask=None,
        cache=None,
        cache_update_index=0,
    ):
        normalized_x = self.pre_attention_norm(x)
        attention_mask = self._compute_attention_mask(
            normalized_x, padding_mask, cache, cache_update_index
        )
        if cache is not None:
            attention, new_cache = self.attention(
                normalized_x,
                attention_mask=attention_mask,
                cache=cache,
                cache_update_index=cache_update_index,
            )
        else:
            attention = self.attention(
                normalized_x,
                attention_mask=attention_mask,
            )

        if self.dropout:
            attention = self.attention_dropout(attention)

        attention_x = x + attention
        normalized_x = self.pre_ffw_norm(attention_x)

        x1 = self.gating_ffw(normalized_x)
        x2 = self.gating_ffw_2(normalized_x)
        x = keras.activations.gelu(x1, approximate=True) * x2
        x = self.ffw_linear(x)

        x = x + attention_x

        if cache is not None:
            return x, new_cache
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "head_dim": self.head_dim,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
            }
        )
        return config
