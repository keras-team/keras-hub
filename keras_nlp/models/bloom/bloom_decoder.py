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
# from keras_nlp.backend import keras
# from keras_nlp.backend import ops
from keras_nlp.api_export import keras_nlp_export
from keras_nlp.backend import keras
from keras_nlp.backend import ops
from keras_nlp.models.bloom.bloom_attention import BloomAttention
from keras_nlp.models.bloom.bloom_mlp import BloomMLP


@keras_nlp_export("keras_nlp.models.BloomDecoder")
class BloomDecoder(keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        dropout=0.0,
        layer_norm_epsilon=1e-5,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        **kwargs,
    ):
        decoder_sequence_shape = kwargs.pop("decoder_sequence_shape", None)

        super().__init__(**kwargs)

        self.num_heads = num_heads
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self._decoder_sequence_shape = None
        if decoder_sequence_shape:
            self.build(decoder_sequence_shape)

    def build(self, decoder_sequence_shape):
        self._decoder_sequence_shape = decoder_sequence_shape

        hidden_dim = decoder_sequence_shape[-1]
        head_dim = int(hidden_dim // self.num_heads)
        if head_dim == 0:
            raise ValueError(
                "Attention `head_dim` computed cannot be zero. "
                f"The `hidden_dim` value of {hidden_dim} has to be equal to "
                f"or greater than `num_heads` value of {self.num_heads}."
            )

        if head_dim * self.num_heads != hidden_dim:
            raise ValueError(
                f"`hidden_dim` must be divisible by num_heads (got `hidden_dim`: {hidden_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self._pre_attention_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="pre_attention_layer_norm",
        )
        self._pre_attention_layernorm.build(decoder_sequence_shape)

        self._self_attention_layer = BloomAttention(
            num_heads=self.num_heads,
            dropout=self.dropout,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            dtype=self.dtype_policy,
            name="self_attention",
        )
        self._self_attention_layer.build(decoder_sequence_shape)

        self._post_attention_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="post_attention_layernorm",
        )
        self._post_attention_layernorm.build(decoder_sequence_shape)

        self._mlp = BloomMLP(
            hidden_dim=hidden_dim,
            dropout=self.dropout,
            dtype=self.dtype_policy,
            name="MLP",
        )
        self._mlp.build(decoder_sequence_shape)

        self.built = True

    def __call__(
        self,
        decoder_sequence,
        **kwargs,
    ):
        if not self.built:
            decoder_sequence_shape = decoder_sequence.shape
            self.build(decoder_sequence_shape)
        return super().__call__(decoder_sequence, **kwargs)

    def call(
        self,
        decoder_sequence,
        decoder_padding_mask=None,
        decoder_attention_mask=None,
        attention_cache=None,
        attention_cache_update_index=None,
        use_causal_mask=True,
    ):
        attention_mask = decoder_attention_mask
        if decoder_attention_mask is None:
            # when `decoder_attention_mask` is passed no need to compute it using
            # `decoder_padding_mask` or using `attention_cache`.
            # otherwise we compute it
            attention_mask = self._compute_attention_mask(
                decoder_sequence=decoder_sequence,
                decoder_padding_mask=decoder_padding_mask,
                attention_cache=attention_cache,
                attention_cache_update_index=attention_cache_update_index,
            )

        if use_causal_mask:
            causal_mask = self._compute_causal_mask(
                decoder_sequence=decoder_sequence,
                attention_cache=attention_cache,
                attention_cache_update_index=attention_cache_update_index,
            )
            attention_mask = ops.minimum(attention_mask, causal_mask)

        residual = decoder_sequence
        x = self._pre_attention_layernorm(decoder_sequence)

        x = self._self_attention_layer(
            x,
            attention_mask,
        )
        x = x + residual

        residual = x
        x = self._post_attention_layernorm(x)
        x = self._mlp(x)
        x = x + residual

        return x

    def _compute_causal_mask(
        self,
        decoder_sequence,
        attention_cache,
        attention_cache_update_index,
    ):
        seq_length = ops.shape(decoder_sequence)[1]
        target_length = source_length = seq_length
        if attention_cache is not None:
            source_length = ops.shape(attention_cache)[2]

        cache_index = (
            0
            if attention_cache_update_index is None
            else attention_cache_update_index
        )

        mask = ops.tri(
            N=target_length, M=source_length, k=cache_index, dtype=int
        )
        return ops.expand_dims(mask, 0)

    def _compute_attention_mask(
        self,
        decoder_sequence,
        decoder_padding_mask,
        attention_cache,
        attention_cache_update_index,
    ):
        seq_length = ops.shape(decoder_sequence)[1]
        target_length = source_length = seq_length

        if decoder_padding_mask is None and attention_cache is None:
            raise ValueError(
                "When "
                "`decoder_padding_mask` and `attention_cache` can't both be None"
            )

        if decoder_padding_mask is not None:
            attention_mask = ops.minimum(
                ops.expand_dims(decoder_padding_mask, 1),
                ops.expand_dims(decoder_padding_mask, -1),
            )

        if attention_cache is not None:
            source_length = ops.shape(attention_cache)[1]
            x = ops.arange(0, source_length)
            x = x < attention_cache_update_index + 1
            x = ops.where(
                x, keras.ones((1), dtype=int), keras.zeros((1), dtype=int)
            )
            x = ops.expand_dims(x, 0)
            x = ops.repeat(x, target_length, axis=0)
            attention_mask = ops.expand_dims(x, 0)

        return attention_mask

    def compute_output_shape(self, decoder_sequence_shape):
        return decoder_sequence_shape


if __name__ == "__main__":
    bd = BloomDecoder(8)
    decode_seq = ops.ones((4, 128, 64))
    bd(decode_seq)
