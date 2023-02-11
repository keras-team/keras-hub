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
from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice


class CachedMultiHeadAttention(keras.layers.MultiHeadAttention):
    """MutliHeadAttention layer with cache support.

    In autoregressive decoding, it's a common practice to cache the K and V for
    previously seen tokens in order to make the computation faster. With cached
    K and V, we can compute the attention output of the last token without
    needing to recompute the forward pass for previously seen tokens. This
    caching method is only useful during decoding, and should not be used
    during training.

    Call arguments:
        query: Query `Tensor` of shape `(B, T, dim)`.
        value: Value `Tensor` of shape `(B, S, dim)`.
        key: Optional key `Tensor` of shape `(B, S, dim)`. If not given, will
            use `value` for both `key` and `value`, which is the most common
            case.
        attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
            attention to certain positions. The boolean mask specifies which
            query elements can attend to which key elements, 1 indicates
            attention and 0 indicates no attention. Broadcasting can happen for
            the missing batch dimensions and the head dimension.
        return_attention_scores: A boolean to indicate whether the output should
            be `(attention_output, attention_scores)` if `True`, or
            `attention_output` if `False`. Defaults to `False`.
        training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (no dropout).
            Defaults to either using the training mode of the parent
            layer/model, or False (inference) if there is no parent layer.
        use_causal_mask: A boolean to indicate whether to apply a causal mask to
            prevent tokens from attending to future tokens (e.g., used in a
            decoder Transformer).
        current_index: a int or int Tensor, defaults to None, the index of the
            current token being processed.
        cache: a dense float Tensor, defaults to None. The cache of key/value of
            leading tokens. `cache` is of shape [2, B, max_seq_len, num_heads,
            key_dims].

    Returns:
        attention_output: The result of the computation, of shape `(B, T, E)`,
            where `T` is for target sequence shapes and `E` is the query input
            last dimension if `output_shape` is `None`. Otherwise, the
            multi-head outputs are projected to the shape specified by
            `output_shape`.
        cache: the updated cache.
        attention_scores: [Optional] multi-head attention coefficients over
            attention axes.
    """

    def call(
        self,
        query,
        value,
        key=None,
        attention_mask=None,
        return_attention_scores=False,
        training=None,
        use_causal_mask=True,
        current_index=None,
        cache=None,
    ):
        if cache is None:
            # When `cache` is None, it's the same as
            # `keras.layers.MultiHeadAttention`.
            return super().call(
                query=query,
                value=value,
                key=key,
                attention_mask=attention_mask,
                return_attention_scores=return_attention_scores,
                training=training,
                use_causal_mask=use_causal_mask,
            )

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

        if current_index is None:
            seq_len = tf.shape(query)[1]
            k_update_indices = [0, 0, 0, 0, 0]
            v_update_indices = [1, 0, 0, 0, 0]
            current_index = seq_len - 1
        else:
            k_update_indices = [0, 0, current_index, 0, 0]
            v_update_indices = [1, 0, current_index, 0, 0]
        cache = dynamic_update_slice(
            cache, key[tf.newaxis, ...], k_update_indices
        )
        cache = dynamic_update_slice(
            cache, value[tf.newaxis, ...], v_update_indices
        )
        keys = cache[0, :, : current_index + 1, :, :]
        values = cache[1, :, : current_index + 1, :, :]
        query = tf.multiply(query, 1.0 / tf.math.sqrt(float(self._key_dim)))
        attention_scores = tf.einsum(self._dot_product_equation, keys, query)
        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )
        attention_scores = self._dropout_layer(attention_scores)

        attention_output = tf.einsum(
            self._combine_equation, attention_scores, values
        )
        attention_output = self._output_dense(attention_output)
        if return_attention_scores:
            return attention_output, cache, attention_scores
        return attention_output, cache
