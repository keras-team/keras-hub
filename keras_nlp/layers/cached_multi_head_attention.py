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

from keras_nlp.api_export import keras_nlp_export


@keras_nlp_export("keras_nlp.layers.CachedMultiHeadAttention")
class CachedMultiHeadAttention(keras.layers.MultiHeadAttention):
    """MutliHeadAttention layer with cache support.

    In autoregressive decoding, it's common practice to cache the key/value
    pairs in both the self-attention layer and the cross-attention layer. In
    the self-attention layer, we cache the key/value pairs of previously seen
    tokens. With cached key and value, we can compute the attention output
    of the last token without recomputing the forward pass for previously seen
    tokens. Secondly, in the cross-attention layer, we cache the key/value
    pairs obtained from the encoder outputs. This way, we only need to do one
    forward pass on the encoder and don't have to recompute the encoder
    key/value pairs for every decoder step. Caching in both the layers makes
    computation faster. This caching method is only useful during decoding, and
    should not be used during training.

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
        cache_update_index: a int or int Tensor, the index of the current token
            being processed. If `cache_update_index=None` while `cache` is set,
            the cache will not be updated.

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
        cache=None,
        cache_update_index=None,
    ):
        if not self._built_from_signature:
            self._build_from_signature(query=query, value=value, key=key)
        if key is None:
            key = value

        query = self._query_dense(query)

        if cache is not None:
            key_cache, value_cache = tf.unstack(cache, axis=1)
            if cache_update_index is not None:
                key_update = self._key_dense(key)
                value_update = self._value_dense(value)
                start = [0, cache_update_index, 0, 0]
                key = dynamic_update_slice(key_cache, key_update, start)
                value = dynamic_update_slice(value_cache, value_update, start)
                cache = tf.stack((key, value), axis=1)
        else:
            key = self._key_dense(key)
            value = self._value_dense(value)

        query = tf.multiply(
            query, 1.0 / tf.math.sqrt(tf.cast(self._key_dim, query.dtype))
        )
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
