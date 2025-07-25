import math

import keras
from keras import ops

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.utils.keras_utils import clone_initializer
from keras_hub.src.utils.keras_utils import fused_attention_op_available


# This is just a self-attention layer in Mistral. But it can be generalized
# to use the `keras_hub.layers.CachedMultiHeadAttention` API. Since this layer
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
        self._inv_norm_factor = 1.0 / math.sqrt(self._head_dim)

        self._query_dense = keras.layers.EinsumDense(
            equation="bqm,muh->bquh",
            output_shape=(None, self._num_query_heads, self._head_dim),
            kernel_initializer=self._kernel_initializer,
            dtype=self.dtype_policy,
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
            dtype=self.dtype_policy,
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
            dtype=self.dtype_policy,
            name="value",
        )
        self._value_dense.build(inputs_shape)

        self._softmax = keras.layers.Softmax(
            axis=-1,
            dtype="float32",
            name="attention_softmax",
        )

        self._dropout_layer = keras.layers.Dropout(
            rate=self._dropout,
            dtype=self.dtype_policy,
        )

        self._output_dense = keras.layers.EinsumDense(
            equation="bquh,uhm->bqm",
            output_shape=(None, self._hidden_dim),
            kernel_initializer=self._kernel_initializer,
            dtype=self.dtype_policy,
            name="attention_output",
        )
        self._output_dense.build(
            (None, None, self._num_query_heads, self._head_dim)
        )

        self.rotary_embedding_layer = RotaryEmbedding(
            max_wavelength=self._rope_max_wavelength,
            scaling_factor=self._rope_scaling_factor,
            dtype=self.dtype_policy,
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
        start_index = (
            cache_update_index if cache_update_index is not None else 0
        )

        query = self._query_dense(hidden_states)

        # Compute RoPE for queries
        query = self.rotary_embedding_layer(query, start_index=start_index)

        def _compute_key_value(x):
            key, value = self._key_dense(x), self._value_dense(x)
            # Compute RoPE for keys
            key = self.rotary_embedding_layer(key, start_index=start_index)
            return key, value

        if cache is not None:
            key_cache = cache[:, 0, ...]
            value_cache = cache[:, 1, ...]
            if cache_update_index is None:
                key = key_cache
                value = value_cache
            else:
                key_update, value_update = _compute_key_value(hidden_states)
                start = [0, cache_update_index, 0, 0]
                key = ops.slice_update(key_cache, start, key_update)
                value = ops.slice_update(value_cache, start, value_update)
                cache = ops.stack((key, value), axis=1)
        else:
            if cache_update_index is not None:
                raise ValueError(
                    "`cache_update_index` should not be set if `cache` is "
                    f"`None`. Received: cache={cache}, "
                    f"cache_update_index={cache_update_index}"
                )
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
        if fused_attention_op_available():
            # Use `dot_product_attention` with Flash Attention support if
            # available.
            if attention_mask is not None:
                attention_mask = ops.expand_dims(attention_mask, axis=1)
                attention_mask = ops.cast(attention_mask, dtype="bool")
            attention_output = ops.dot_product_attention(
                query,
                key,
                value,
                mask=attention_mask,
                scale=self._inv_norm_factor,
            )
            return attention_output

        attention_scores = ops.einsum(self._dot_product_equation, query, key)
        attention_scores = ops.multiply(
            attention_scores,
            ops.cast(self._inv_norm_factor, self.compute_dtype),
        )
        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )
        attention_scores = ops.cast(attention_scores, self.compute_dtype)
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
