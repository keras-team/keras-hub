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
"""Cached MHA layer based on `keras.layers.MultiHeadAttention`."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice


class CachedMultiHeadAttention(keras.layers.MultiHeadAttention):
    """MutliHeadAttention layer with cache support.

    In autoregressive decoding, it's a common practice to cache the key/value in
    multi-head attention of previously seen tokens in order to make the
    computation faster. With cached K and V, we can compute the attention output
    of the last token without recomputing the forward pass for previously seen
    tokens. This caching method is only useful during decoding, and should not
    be used during training.

    Call arguments:
        query: Query `Tensor` of shape `(B, T, dim)` if `cache=None`,
            otherwise `(B, 1, dim)`.
        value: Value `Tensor` of shape `(B, S, dim)` if `cache=None`,
            otherwise `(B, 1, dim)`.
        key: Optional key `Tensor` of shape `(B, S, dim)` if `cache=None`,
            otherwise `(B, 1, dim)`. If not given, will use `value` for both
            `key` and `value`, which is the most common case.
        attention_mask: a boolean mask of shape `(B, T, S)` if `cache=None`,
            otherwise `(B, 1, S)`. `attention_mask` prevents
            attention to certain positions. The boolean mask specifies which
            query elements can attend to which key elements, 1 indicates
            attention and 0 indicates no attention. Broadcasting can happen for
            the missing batch dimensions and the head dimension.
        cache: a dense float Tensor. The cache of key/value of leading tokens.
            `cache` is of shape [B, 2, max_seq_len, num_heads, key_dims].
        cache_index: a int or int Tensor, the index of the current token being
            processed. If `cache_index=None` while `cache` is set, it means
            it's the first pass to build the cache.

    Returns:
        An (attention_output, cache) tuple. `attention_output` is the result of
        the computation, of shape `(B, T, E)`, where `T` is for target sequence
        shapes and `E` is the query input last dimension if  `output_shape` is
        `None`. Otherwise, the multi-head outputs are projected to the shape
        specified by `output_shape`. `cache` is the updated cache.
    """

    def call(
        self,
        query,
        value,
        key=None,
        attention_mask=None,
        use_causal_mask=False,
        cache=None,
        cache_index=None,
    ):
        if not self._built_from_signature:
            self._build_from_signature(query=query, value=value, key=key)
        if key is None:
            key = value

        attention_mask = self._compute_attention_mask(
            query,
            value,
            key=key,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
        )

        query = self._query_dense(query)
        key = self._key_dense(key)
        value = self._value_dense(value)

        if cache is not None:
            if cache_index is None:
                seq_len = tf.shape(query)[1]
                k_update_indices = [0, 0, 0, 0, 0]
                v_update_indices = [0, 1, 0, 0, 0]
                cache_index = seq_len - 1
            else:
                k_update_indices = [0, 0, cache_index, 0, 0]
                v_update_indices = [0, 1, cache_index, 0, 0]
            cache = dynamic_update_slice(
                cache, key[:, tf.newaxis, ...], k_update_indices
            )
            cache = dynamic_update_slice(
                cache, value[:, tf.newaxis, ...], v_update_indices
            )
            key = cache[:, 0, : cache_index + 1, :, :]
            value = cache[:, 1, : cache_index + 1, :, :]

        query = tf.multiply(query, 1.0 / tf.math.sqrt(float(self._key_dim)))
        attention_scores = tf.einsum(self._dot_product_equation, key, query)
        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )
        attention_scores = self._dropout_layer(attention_scores)

        attention_output = tf.einsum(
            self._combine_equation, attention_scores, value
        )
        attention_output = self._output_dense(attention_output)
        return attention_output, cache
