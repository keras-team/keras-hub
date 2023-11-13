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
from keras_nlp.backend import keras
from keras_nlp.backend import ops
from keras_nlp.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_nlp.utils.keras_utils import clone_initializer


# This is just a self-attention layer in Mistral. But it can be generalized
# to use the `keras_nlp.layers.CachedMultiHeadAttention` API. Since this layer
# implements grouped-query attention and sliding window attention, it might be
# useful outside of Mistral itself.
# TODO(tirthasheshpatel): Generalize the attention layer
# TODO(tirthasheshpatel): Merge `LlamaAttention` with this layer
# TODO(tirthasheshpatel): Use flash attention
# TODO(tirthasheshpatel): Add dropout
class CachedMistralAttention(keras.layers.Layer):
    def __init__(
        self,
        num_query_heads,
        num_key_value_heads,
        rope_max_wavelength=10000,
        rope_scaling_factor=1.0,
        kernel_initializer="glorot_uniform",
        sliding_window=512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._num_query_heads = num_query_heads
        self._num_key_value_heads = num_key_value_heads
        self._sliding_window = sliding_window

        self._num_key_value_groups = num_query_heads // num_key_value_heads
        self._rope_max_wavelength = rope_max_wavelength

        self._kernel_initializer = keras.initializers.get(
            clone_initializer(kernel_initializer)
        )

        self._rope_scaling_factor = rope_scaling_factor

    def build(self, inputs_shape):
        self._hidden_dim = inputs_shape[-1]
        self._attn_head_size = self._hidden_dim // self._num_query_heads

        self._query_dense = keras.layers.EinsumDense(
            equation="abc,cde->abde",
            output_shape=(None, self._num_query_heads, self._attn_head_size),
            kernel_initializer=self._kernel_initializer,
            name="query",
        )
        self._query_dense.build(inputs_shape)

        self._key_dense = keras.layers.EinsumDense(
            equation="abc,cde->abde",
            output_shape=(
                None,
                self._num_key_value_heads,
                self._attn_head_size,
            ),
            kernel_initializer=self._kernel_initializer,
            name="key",
        )
        self._key_dense.build(inputs_shape)

        self._value_dense = keras.layers.EinsumDense(
            equation="abc,cde->abde",
            output_shape=(
                None,
                self._num_key_value_heads,
                self._attn_head_size,
            ),
            kernel_initializer=self._kernel_initializer,
            name="value",
        )
        self._value_dense.build(inputs_shape)

        self._softmax = keras.layers.Softmax(axis=-1, name="attention_softmax")

        self._output_dense = keras.layers.EinsumDense(
            equation="abc,cd->abd",
            output_shape=(None, self._hidden_dim),
            kernel_initializer=self._kernel_initializer,
            name="attention_output",
        )
        self._output_dense.build(inputs_shape)

        self.rotary_embedding_layer = RotaryEmbedding(
            max_wavelength=self._rope_max_wavelength,
            scaling_factor=self._rope_scaling_factor,
        )

        self.built = True

    def call(
        self,
        hidden_states,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
    ):
        seq_len = ops.shape(hidden_states)[1]
        start_index = (
            cache_update_index if cache_update_index is not None else 0
        )

        query = self._query_dense(hidden_states)

        # Note that the original PyTorch implementation uses
        # view_as_complex/view_as_real while we use split/concatenate to
        # convert to/from complex numbers. The transformations below make
        # the rope computation numerically equivalent to the original
        # implementation.
        def _mistral_rope(x):
            x = ops.concatenate([x[..., ::2], x[..., 1::2]], axis=-1)
            x = self.rotary_embedding_layer(x, start_index=start_index)
            x = ops.reshape(
                ops.stack(ops.split(x, 2, axis=-1), axis=-1), ops.shape(x)
            )
            return x

        # Compute RoPE for queries
        query = _mistral_rope(query)

        # Note that the cache update step is slightly different for the
        # mistral model. If `seq_len` is greater than `self._sliding_window`,
        # the cache wraps around itself. So, in this case, we update all the
        # indices as we would, but at the end, we would permute the cache such
        # that the tokens outside the sliding window appear in front of the
        # cache.
        def _update_cache(cache, update_positions, update):
            cache = ops.slice_update(
                cache, [0, ops.min(update_positions), 0, 0], update
            )
            perm = ops.concatenate(
                [
                    ops.arange(0, ops.min(update_positions)),
                    update_positions,
                    ops.arange(
                        ops.max(update_positions) + 1, self._sliding_window
                    ),
                ],
                axis=-1,
            )
            cache = ops.take(cache, perm, axis=1)
            return cache

        def _compute_key_value(x):
            key, value = self._key_dense(x), self._value_dense(x)
            key = _mistral_rope(key)
            return key, value

        if cache is not None:
            cache_k = cache[:, 0, ...]
            cache_v = cache[:, 1, ...]

            if cache_update_index is not None:
                # Compute the new keys and values
                key, value = _compute_key_value(hidden_states)

                # Cache is a rotating buffer
                positions = ops.arange(
                    cache_update_index, cache_update_index + seq_len
                )
                update_positions = (
                    positions[-self._sliding_window :] % self._sliding_window
                )
                cache_k = _update_cache(
                    cache_k,
                    update_positions,
                    ops.cast(
                        key[:, -self._sliding_window :, ...], cache_k.dtype
                    ),
                )
                cache_v = _update_cache(
                    cache_v,
                    update_positions,
                    ops.cast(
                        value[:, -self._sliding_window :, ...], cache_v.dtype
                    ),
                )
                cache = ops.stack([cache_k, cache_v], axis=1)

            # Get the required keys and values from the cache.
            # Since we expect the user to pass a fixed-size cache, we just
            # pick the first few slices up-to and including the newly computed
            # keys and values.
            key = ops.cast(
                cache_k[
                    :,
                    : (cache_update_index + seq_len - 1) % self._sliding_window
                    + 1,
                    ...,
                ],
                dtype=self.compute_dtype,
            )
            value = ops.cast(
                cache_v[
                    :,
                    : (cache_update_index + seq_len - 1) % self._sliding_window
                    + 1,
                    ...,
                ],
                dtype=self.compute_dtype,
            )
        else:
            # Compute keys and values
            key, value = _compute_key_value(hidden_states)

        # [batch_shape, seq_len, num_key_value_heads, head_dim]
        # -> [batch_shape, seq_len, num_heads, head_dim]
        key = ops.repeat(key, repeats=self._num_key_value_groups, axis=2)
        value = ops.repeat(value, repeats=self._num_key_value_groups, axis=2)

        attention_output = self._compute_attention(
            query, key, value, attention_mask
        )

        attention_output_shape = ops.shape(attention_output)
        attention_output = ops.reshape(
            attention_output,
            [
                attention_output_shape[0],  # batch_shape
                attention_output_shape[1],  # seq_len
                self._hidden_dim,
            ],
        )

        attention_output = self._output_dense(attention_output)

        if cache is not None:
            return attention_output, cache
        return attention_output

    def _masked_softmax(self, attention_scores, attention_mask=None):
        if attention_mask is not None:
            return self._softmax(
                attention_scores, attention_mask[:, None, :, :]
            )
        return self._softmax(attention_scores)

    def _compute_attention(self, query, key, value, attention_mask=None):
        attention_scores = ops.einsum("aecd,abcd->acbe", key, query)

        norm_factor = ops.sqrt(
            ops.cast(self._attn_head_size, self.compute_dtype)
        )

        attention_scores = attention_scores / norm_factor

        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )
        attention_output = ops.einsum(
            "acbe,aecd->abcd", attention_scores, value
        )

        return attention_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_query_heads": self._num_query_heads,
                "num_key_value_heads": self._num_key_value_heads,
                "rope_max_wavelength": self._rope_max_wavelength,
                "rope_scaling_factor": self._rope_scaling_factor,
                "kernel_initializer": keras.initializers.serialize(
                    self._kernel_initializer
                ),
                "sliding_window": self._sliding_window,
            }
        )
        return config
