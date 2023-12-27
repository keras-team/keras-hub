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


class BloomAttention(keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        dropout=0.0,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.dropout = dropout
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.inv_norm_factor = None

    def build(self, inputs_shape):
        _, seq_length, hidden_dim = inputs_shape

        self.head_dim = hidden_dim // self.num_heads

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)

        self._query_key_value_dense = keras.layers.Dense(
            3 * hidden_dim,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            dtype=self.dtype_policy,
            name="query_key_value_dense",
        )
        self._query_key_value_dense.build(inputs_shape)

        self._out_dense = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            dtype=self.dtype_policy,
            name="out_dense",
        )
        self._out_dense.build(inputs_shape)

        self._dropout = keras.layers.Dropout(
            rate=self.dropout, dtype=self.dtype_policy, name="dropout"
        )
        self._softmax = keras.layers.Softmax(
            dtype=self.dtype_policy, name="softmax"
        )

        self.built = True

    def _split_heads(self, fused_qkv):
        # in  : [batch_size, seq_length, 3 x hidden_size]
        # out : 3 * [batch_size, num_heads, seq_length, head_dim]
        batch_size, seq_length, _ = ops.shape(fused_qkv)
        fused_qkv = ops.reshape(
            fused_qkv,
            (batch_size, seq_length, self.num_heads, 3, self.head_dim),
        )
        return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]

    @staticmethod
    def _build_alibi_tensor(seq_length, num_heads):
        def get_slopes(n):
            # this function is adopted from fairseq
            # https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][
                        : n - closest_power_of_2
                    ]
                )

        slopes = ops.convert_to_tensor(get_slopes(num_heads), dtype=float)
        slopes = ops.expand_dims(slopes, 1)

        alibi = slopes * ops.expand_dims(
            ops.arange(seq_length, dtype=float), 0
        )
        alibi = ops.expand_dims(alibi, 1)
        alibi = ops.expand_dims(alibi, 0)

        return alibi

    def call(
        self,
        hidden_states,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
    ):
        batch_size, seq_length, _ = ops.shape(hidden_states)

        fused_qkv = self._query_key_value_dense(
            hidden_states
        )  # [batch_size, seq_length, 3 x hidden_size]

        (query, key, value) = self._split_heads(
            fused_qkv
        )  # [batch_size, seq_length, num_heads, head_dim]

        if cache is not None:
            key_cache = cache[:, 0, ...]
            value_cache = cache[:, 1, ...]
            if cache_update_index is None:
                key = key_cache
                value = value_cache
            else:
                start = [0, cache_update_index, 0, 0]
                key = ops.slice_update(key_cache, start, key)
                value = ops.slice_update(value_cache, start, value)
                cache = ops.stack((key, value), axis=1)
        else:
            if cache_update_index is not None:
                raise ValueError(
                    "`cache_update_index` should not be set if `cache` is "
                    f"`None`. Received: cache={cache}, "
                    f"cache_update_index={cache_update_index}"
                )

        # query (batch_size, num_heads, query_length, head_dim)
        query = ops.transpose(query, [0, 2, 1, 3])
        # value (batch_size, num_heads, kv_length, head_dim)
        value = ops.transpose(value, [0, 2, 1, 3])
        # key   (batch_size, num_heads, head_dim, kv_length)
        key = ops.transpose(key, [0, 2, 3, 1])

        alibi = self._build_alibi_tensor(
            seq_length=seq_length, num_heads=self.num_heads
        )

        scores = (
            ops.matmul(query, key) * self.inv_norm_factor + alibi
        )  # [batch_size, num_heads, query_length, kv_length]

        scores = self._softmax(scores, ops.expand_dims(attention_mask, 1))

        scores = self._dropout(scores)

        attention_output = ops.matmul(
            scores, value
        )  # [batch_size, num_heads, query_length, head_dim]

        attention_output = ops.transpose(
            attention_output, [0, 2, 1, 3]
        )  # [batch_size, query_length, num_heads, head_dim]
        attention_output = ops.reshape(
            attention_output,
            [batch_size, seq_length, self.num_heads * self.head_dim],
        )  # [batch_size, query_length, hidden_dim]

        attention_output = self._out_dense(attention_output)
        attention_output = self._dropout(attention_output)

        if cache is not None:
            return attention_output, cache

        return attention_output
