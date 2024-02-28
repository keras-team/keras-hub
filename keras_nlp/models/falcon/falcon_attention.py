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
import math

from keras_nlp.backend import keras
from keras_nlp.backend import ops


class FalconAttention(keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        attention_dropout,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout

    def build(self, inputs_shape):
        batch_size, seq_length, hidden_dim = inputs_shape

        self.head_dim = hidden_dim // self.num_heads

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)

        self._query_dense = keras.layers.EinsumDense(
            equation="btm,mnh->btnh",
            output_shape=(None, self.num_heads, self.head_dim),
            bias_axes="nh",
            dtype=self.dtype_policy,
            name="query_dense",
        )
        self._query_dense.build(inputs_shape)

        self._key_dense = keras.layers.EinsumDense(
            equation="bsm,mnh->bsnh",
            output_shape=(None, self.num_heads, self.head_dim),
            bias_axes="nh",
            dtype=self.dtype_policy,
            name="key_dense",
        )
        self._key_dense.build(inputs_shape)

        self._value_dense = keras.layers.EinsumDense(
            equation="bsm,mnh->bsnh",
            output_shape=(None, self.num_heads, self.head_dim),
            bias_axes="nh",
            dtype=self.dtype_policy,
            name="value_dense",
        )
        self._value_dense.build(inputs_shape)

        self._attention_dropout = keras.layers.Dropout(
            rate=self.attention_dropout,
            dtype=self.dtype_policy,
            name="attention_dropout",
        )

        self._output_dense = keras.layers.Dense(
            hidden_dim,
            dtype=self.dtype_policy,
            name="output_dense",
        )
        self._output_dense.build(inputs_shape)

        self._softmax = keras.layers.Softmax(dtype="float32", name="softmax")

        self.built = True

    def call(
        self,
        inputs,
        alibi,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
    ):
        batch_size, seq_length, hidden_dim = ops.shape(inputs)

        query = self._query_dense(inputs)
        key = self._key_dense(inputs)
        value = self._value_dense(inputs)

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

        attention_scores = ops.matmul(query, key)
        attention_scores = ops.add(attention_scores, alibi)
        attention_scores = (
            attention_scores * self.inv_norm_factor
        )  # [batch_size, num_heads, query_length, kv_length]
        attention_scores = self._softmax(
            attention_scores, ops.expand_dims(attention_mask, 1)
        )
        attention_scores = self._attention_dropout(attention_scores)
        attention_output = ops.matmul(
            attention_scores, value
        )  # [batch_size, num_heads, query_length, head_dim]

        attention_output = ops.transpose(
            attention_output, [0, 2, 1, 3]
        )  # [batch_size, query_length, num_heads, head_dim]
        attention_output = ops.reshape(
            attention_output,
            [batch_size, seq_length, self.num_heads * self.head_dim],
        )  # [batch_size, query_length, hidden_dim]

        attention_output = self._output_dense(attention_output)

        if cache is not None:
            return attention_output, cache

        return attention_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "attention_dropout": self.attention_dropout,
            }
        )
        return config
