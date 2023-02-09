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
    """

    def _update_cache(self, key, value, cache):
        """Updates cache states and gets full-length key/value tensors."""
        if "keys" not in cache:
            keys = key
            values = value
        else:
            keys = tf.concat([tf.cast(cache["keys"], key.dtype), key], axis=1)
            values = tf.concat(
                [tf.cast(cache["values"], value.dtype), value], axis=1
            )

        # Update cache
        cache["keys"] = keys
        cache["values"] = values

        return keys, values

    def call(
        self,
        query,
        value,
        key=None,
        current_index=None,
        attention_mask=None,
        cache=None,
        return_attention_scores=False,
        training=None,
        use_causal_mask=True,
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
        cache = dynamic_update_slice(cache, [key], k_update_indices)
        cache = dynamic_update_slice(cache, [value], v_update_indices)
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
        return attention_output, cache
