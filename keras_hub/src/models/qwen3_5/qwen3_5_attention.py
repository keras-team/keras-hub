import math

import keras
from keras import ops

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.models.qwen3_5.qwen3_5_layernorm import Qwen3_5LayerNorm
from keras_hub.src.utils.keras_utils import clone_initializer
from keras_hub.src.utils.keras_utils import fused_attention_op_available


class Qwen3_5Attention(keras.layers.Layer):
    """Full self-attention layer for Qwen3.5.

    This implements grouped-query attention (GQA) with:
    - Q/K RMSNorm
    - Partial rotary embeddings (only first `partial_rotary_factor` fraction
      of head_dim gets RoPE)
    - Sigmoid gating on attention output
    - Optional sliding window

    Args:
        num_query_heads: Number of query attention heads.
        num_key_value_heads: Number of key/value attention heads (GQA).
        head_dim: Dimension of each attention head.
        partial_rotary_factor: Fraction of head_dim that gets RoPE.
        rope_max_wavelength: Maximum wavelength for rotary embeddings.
        rope_scaling_factor: Scaling factor for rotary embeddings.
        kernel_initializer: Initializer for projection kernels.
        dropout: Dropout rate for attention weights.
        layer_norm_epsilon: Epsilon for Q/K RMSNorm.
        sliding_window_size: Optional sliding window size.
    """

    def __init__(
        self,
        num_query_heads,
        num_key_value_heads,
        head_dim,
        partial_rotary_factor=0.25,
        rope_max_wavelength=10000,
        rope_scaling_factor=1.0,
        kernel_initializer="glorot_uniform",
        dropout=0.0,
        layer_norm_epsilon=1e-6,
        sliding_window_size=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.partial_rotary_factor = partial_rotary_factor
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.num_key_value_groups = num_query_heads // num_key_value_heads
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.sliding_window_size = sliding_window_size
        self.kernel_initializer = keras.initializers.get(
            clone_initializer(kernel_initializer)
        )

    def build(self, inputs_shape):
        hidden_dim = inputs_shape[-1]
        self._inv_norm_factor = 1.0 / math.sqrt(self.head_dim)

        # Q projects to (num_query_heads, head_dim * 2) to include gate.
        self._query_dense = keras.layers.EinsumDense(
            equation="bqm,muh->bquh",
            output_shape=(
                None,
                self.num_query_heads,
                self.head_dim * 2,
            ),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="query",
        )
        self._query_dense.build(inputs_shape)

        self._query_norm = Qwen3_5LayerNorm(
            head_dim=self.head_dim,
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="query_norm",
        )
        self._query_norm.build(
            (None, None, self.num_query_heads, self.head_dim)
        )

        self._key_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(
                None,
                self.num_key_value_heads,
                self.head_dim,
            ),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="key",
        )
        self._key_dense.build(inputs_shape)

        self._key_norm = Qwen3_5LayerNorm(
            head_dim=self.head_dim,
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="key_norm",
        )
        self._key_norm.build(
            (None, None, self.num_key_value_heads, self.head_dim)
        )

        self._value_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(
                None,
                self.num_key_value_heads,
                self.head_dim,
            ),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="value",
        )
        self._value_dense.build(inputs_shape)

        self._softmax = keras.layers.Softmax(
            axis=-1, dtype="float32", name="attention_softmax"
        )
        self._dropout_layer = keras.layers.Dropout(
            rate=self.dropout, dtype=self.dtype_policy
        )
        self._output_dense = keras.layers.EinsumDense(
            equation="bquh,uhm->bqm",
            output_shape=(None, hidden_dim),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="attention_output",
        )
        self._output_dense.build(
            (None, None, self.num_query_heads, self.head_dim)
        )

        self.rotary_embedding_layer = RotaryEmbedding(
            max_wavelength=self.rope_max_wavelength,
            scaling_factor=self.rope_scaling_factor,
            dtype=self.dtype_policy,
        )

        self._dot_product_equation = "bquh,bkuh->buqk"
        self._combine_equation = "buqk,bkuh->bquh"
        self.built = True

    def _apply_partial_rope(self, x, start_index):
        """Apply RoPE only to the first `rotary_dim` dimensions."""
        if self.rotary_dim == self.head_dim:
            return self.rotary_embedding_layer(x, start_index=start_index)

        x_rope = x[..., : self.rotary_dim]
        x_pass = x[..., self.rotary_dim :]
        x_rope = self.rotary_embedding_layer(x_rope, start_index=start_index)
        return ops.concatenate([x_rope, x_pass], axis=-1)

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

        # Query projects to (head_dim * 2), split into query + gate.
        qg = self._query_dense(hidden_states)
        query = qg[..., : self.head_dim]
        gate = qg[..., self.head_dim :]

        # Reshape gate for per-head gating: (B, seq, heads * head_dim)
        gate_shape = ops.shape(gate)
        gate = ops.reshape(
            gate,
            (gate_shape[0], gate_shape[1], -1),
        )

        query = self._query_norm(query)
        query = self._apply_partial_rope(query, start_index)

        def _compute_key_value(x):
            key = self._key_dense(x)
            key = self._key_norm(key)
            key = self._apply_partial_rope(key, start_index)
            value = self._value_dense(x)
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
                    "`cache_update_index` should not be set if `cache` "
                    f"is `None`. Received: cache={cache}, "
                    f"cache_update_index={cache_update_index}"
                )
            key, value = _compute_key_value(hidden_states)

        # GQA: repeat K/V heads.
        key = ops.repeat(key, repeats=self.num_key_value_groups, axis=2)
        value = ops.repeat(value, repeats=self.num_key_value_groups, axis=2)

        attention_output = self._compute_attention(
            query,
            key,
            value,
            attention_mask,
            cache_update_index=cache_update_index,
        )
        attention_output = self._dropout_layer(
            attention_output, training=training
        )

        # Reshape to (B, seq, heads * head_dim) for gating.
        out_shape = ops.shape(attention_output)
        attention_output = ops.reshape(
            attention_output,
            (out_shape[0], out_shape[1], -1),
        )

        # Apply sigmoid gate.
        attention_output = attention_output * ops.sigmoid(gate)

        # Reshape back to (B, seq, heads, head_dim) for output proj.
        attention_output = ops.reshape(
            attention_output,
            (out_shape[0], out_shape[1], self.num_query_heads, self.head_dim),
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

    def _compute_attention(
        self,
        query,
        key,
        value,
        attention_mask=None,
        cache_update_index=None,
    ):
        if fused_attention_op_available():
            if attention_mask is not None:
                attention_mask = ops.expand_dims(attention_mask, axis=1)
                attention_mask = ops.cast(attention_mask, dtype="bool")
            return ops.dot_product_attention(
                query,
                key,
                value,
                mask=attention_mask,
                scale=self._inv_norm_factor,
            )

        attention_scores = ops.einsum(self._dot_product_equation, query, key)
        attention_scores = ops.multiply(
            attention_scores,
            ops.cast(self._inv_norm_factor, self.compute_dtype),
        )
        if self.sliding_window_size:
            attention_mask = self._mask_sliding_window(
                attention_mask,
                cache_update_index=(
                    cache_update_index if cache_update_index is not None else 0
                ),
            )
        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )
        attention_scores = ops.cast(attention_scores, self.compute_dtype)
        return ops.einsum(self._combine_equation, attention_scores, value)

    def _mask_sliding_window(self, attention_mask, cache_update_index=0):
        _, query_len, key_len = ops.shape(attention_mask)
        all_ones = ops.ones((key_len, key_len), "bool")
        if keras.config.backend() == "tensorflow":
            import tensorflow as tf

            band_size = ops.minimum(key_len, self.sliding_window_size - 1)
            band_size = ops.cast(band_size, "int32")
            sliding_mask = tf.linalg.band_part(all_ones, band_size, band_size)
        else:
            sliding_mask = ops.triu(
                all_ones, -1 * self.sliding_window_size + 1
            ) * ops.tril(all_ones, self.sliding_window_size - 1)
        start = (cache_update_index, 0)
        sliding_mask = ops.slice(sliding_mask, start, (query_len, key_len))
        sliding_mask = ops.expand_dims(sliding_mask, 0)
        return ops.logical_and(attention_mask, ops.cast(sliding_mask, "bool"))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
                "partial_rotary_factor": self.partial_rotary_factor,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "dropout": self.dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "sliding_window_size": self.sliding_window_size,
            }
        )
        return config
