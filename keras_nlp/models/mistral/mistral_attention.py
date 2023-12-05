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
class CachedMistralAttention(keras.layers.Layer):
    """A cached grounded query attention layer with sliding window."""

    def __init__(
        self,
        num_query_heads,
        num_key_value_heads,
        rope_max_wavelength=10000,
        rope_scaling_factor=1.0,
        kernel_initializer="glorot_uniform",
        sliding_window=512,
        dropout=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._num_query_heads = num_query_heads
        self._num_key_value_heads = num_key_value_heads
        self._sliding_window = sliding_window
        self._dropout = dropout

        self._num_key_value_groups = num_query_heads // num_key_value_heads
        self._rope_max_wavelength = rope_max_wavelength

        self._kernel_initializer = keras.initializers.get(
            clone_initializer(kernel_initializer)
        )

        self._rope_scaling_factor = rope_scaling_factor

    def build(self, inputs_shape):
        # Einsum variables:
        # b = batch size
        # q = query length
        # k = key/value length
        # m = model dim
        # u = num query heads
        # v = num key/value heads
        # h = head dim
        self._hidden_dim = inputs_shape[-1]
        self._head_dim = self._hidden_dim // self._num_query_heads

        self._query_dense = keras.layers.EinsumDense(
            equation="bqm,muh->bquh",
            output_shape=(None, self._num_query_heads, self._head_dim),
            kernel_initializer=self._kernel_initializer,
            dtype=self.compute_dtype,
            name="query",
        )
        self._query_dense.build(inputs_shape)

        self._key_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(
                None,
                self._num_key_value_heads,
                self._head_dim,
            ),
            kernel_initializer=self._kernel_initializer,
            dtype=self.compute_dtype,
            name="key",
        )
        self._key_dense.build(inputs_shape)

        self._value_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(
                None,
                self._num_key_value_heads,
                self._head_dim,
            ),
            kernel_initializer=self._kernel_initializer,
            dtype=self.compute_dtype,
            name="value",
        )
        self._value_dense.build(inputs_shape)

        self._softmax = keras.layers.Softmax(axis=-1, name="attention_softmax")

        self._dropout_layer = keras.layers.Dropout(
            rate=self._dropout, dtype=self.compute_dtype
        )

        self._output_dense = keras.layers.EinsumDense(
            equation="bquh,uhm->bqm",
            output_shape=(None, self._hidden_dim),
            kernel_initializer=self._kernel_initializer,
            dtype=self.compute_dtype,
            name="attention_output",
        )
        self._output_dense.build(
            (None, None, self._num_query_heads, self._head_dim)
        )

        self.rotary_embedding_layer = RotaryEmbedding(
            max_wavelength=self._rope_max_wavelength,
            scaling_factor=self._rope_scaling_factor,
            dtype=self.compute_dtype,
        )

        self._dot_product_equation = "bquh,bkuh->buqk"
        self._combine_equation = "buqk,bkuh->bquh"

        self.built = True

    def call(
        self,
        hidden_states,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
        training=None,
    ):
        seq_len = ops.shape(hidden_states)[1]
        start_index = (
            cache_update_index if cache_update_index is not None else 0
        )
        # If `cache_update_index` is a tensor, RotaryEmbedding expects it
        # to have dtype `self.compute_dtype`.
        start_index = ops.cast(
            start_index, self.rotary_embedding_layer.compute_dtype
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

                # Cache is a rotating buffer, we want to warp around if
                # the sequence length exceeds the sliding window.
                update_end_index = (
                    cache_update_index + seq_len - 1
                ) % self._sliding_window + 1
                update_end_index = ops.cast(update_end_index, "int32")
                cache_update_index = cache_update_index % self._sliding_window
                update_start_index = ops.cond(
                    update_end_index > cache_update_index,
                    lambda: ops.cast(cache_update_index, "int32"),
                    lambda: ops.cast(0, "int32"),
                )
                # Also note that the update step below assumes that the
                # sequence length is always one when `cache_update_index != 0`.
                # This is necessary to support XLA compilation. Ideally, we
                # would want to use
                # `key[:, -(update_end_index - update_start_index):, ...]`
                # as the update but updating using a dynamic slice gives an
                # XLA compilation error in TensorFlow.
                # Passing a sequence of length > 1 with cache update might give
                # incorrect results (since there is no way to determine how
                # many most recent tokens are to be saved if the tokens exceed
                # the sliding window length).
                cache_k = ops.slice_update(
                    cache_k,
                    [0, update_start_index, 0, 0],
                    # We slice the keys and values since if the user has passed
                    # a sequence of length > `self._sliding_window`. We want to
                    # prefill the cache using just the most recent values in the
                    # sliding window.
                    ops.cast(
                        key[:, -self._sliding_window :, ...], cache_k.dtype
                    ),
                )
                cache_v = ops.slice_update(
                    cache_v,
                    [0, update_start_index, 0, 0],
                    ops.cast(
                        value[:, -self._sliding_window :, ...], cache_v.dtype
                    ),
                )
                cache = ops.stack([cache_k, cache_v], axis=1)

                # Get the required keys and values from the cache.
                # Since we expect the user to pass a fixed-size cache, we just
                # pick the first few slices up-to and including the newly computed
                # keys and values.
                cache_k = cache_k[:, :update_end_index, ...]
                cache_v = cache_v[:, :update_end_index, ...]

            key = ops.cast(cache_k, dtype=self.compute_dtype)
            value = ops.cast(cache_v, dtype=self.compute_dtype)
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

        attention_output = self._dropout_layer(
            attention_output, training=training
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
        attention_scores = ops.einsum(self._dot_product_equation, key, query)

        norm_factor = ops.sqrt(ops.cast(self._head_dim, self.compute_dtype))

        attention_scores = attention_scores / norm_factor

        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )
        attention_output = ops.einsum(
            self._combine_equation, attention_scores, value
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
                "dropout": self._dropout,
            }
        )
        return config
