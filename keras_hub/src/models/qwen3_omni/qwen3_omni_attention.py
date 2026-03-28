import math

import keras
from keras import ops

from keras_hub.src.models.qwen3_moe.qwen3_moe_layernorm import Qwen3MoeLayerNorm
from keras_hub.src.models.qwen3_omni.qwen3_omni_rope import (
    MultimodalRotaryEmbedding,
)
from keras_hub.src.utils.keras_utils import clone_initializer
from keras_hub.src.utils.keras_utils import fused_attention_op_available


class Qwen3OmniAttention(keras.layers.Layer):
    """Multi-head attention with Multimodal RoPE for Qwen3-Omni.

    This attention layer implements:
    - Grouped Query Attention (GQA) for efficiency
    - Multimodal Rotary Position Embedding (M-RoPE) for multimodal inputs
    - Query-Key normalization for training stability
    - Optional sliding window attention

    The M-RoPE divides the head dimension into 3 sections (24, 20, 20) for
    text, temporal, and spatial position encodings respectively.

    Args:
        num_query_heads: int. Number of query heads.
        num_key_value_heads: int. Number of key/value heads (for GQA).
        head_dim: int. The dimension of each attention head.
        mrope_section: tuple of 3 ints. Dimension allocation for M-RoPE
            (text, temporal, spatial). Defaults to (24, 20, 20).
        rope_max_wavelength: int. Maximum wavelength for M-RoPE.
            Defaults to 1000000.
        rope_scaling_factor: float. Scaling factor for M-RoPE. Defaults to 1.0.
        kernel_initializer: Initializer for kernel weights.
        dropout: float. Dropout rate for attention weights.
        layer_norm_epsilon: float. Epsilon for layer normalization.
        sliding_window_size: int or None. Size of sliding window.
            Defaults to None.
        **kwargs: Additional keyword arguments to pass to the layer
    """

    def __init__(
        self,
        num_query_heads,
        num_key_value_heads,
        head_dim=None,
        mrope_section=(24, 20, 20),
        rope_max_wavelength=1000000,
        rope_scaling_factor=1.0,
        rope_attention_scaling=1.0,
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
        self.mrope_section = tuple(mrope_section)
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.num_key_value_groups = num_query_heads // num_key_value_heads
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_attention_scaling = rope_attention_scaling
        self.kernel_initializer = keras.initializers.get(
            clone_initializer(kernel_initializer)
        )
        self.sliding_window_size = sliding_window_size

    def build(self, inputs_shape):
        # Einsum variables:
        # b = batch size
        # q = query length
        # k = key/value length
        # m = model dim
        # u = num query heads
        # v = num key/value heads
        # h = head dim
        hidden_dim = inputs_shape[-1]
        if not self.head_dim:
            self.head_dim = hidden_dim // self.num_query_heads

        self._inv_norm_factor = 1.0 / math.sqrt(self.head_dim)

        # Query projection with EinsumDense
        self._query_dense = keras.layers.EinsumDense(
            equation="bqm,muh->bquh",
            output_shape=(None, self.num_query_heads, self.head_dim),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="query",
        )
        self._query_dense.build(inputs_shape)

        # Query normalization (QK norm)
        self._query_dense_layer_norm = Qwen3MoeLayerNorm(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            head_dim=self.head_dim,
            name="query_dense_layernorm",
        )
        self._query_dense_layer_norm.build(inputs_shape)

        # Key projection
        self._key_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="key",
        )
        self._key_dense.build(inputs_shape)

        # Key normalization (QK norm)
        self._key_dense_layer_norm = Qwen3MoeLayerNorm(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            head_dim=self.head_dim,
            name="key_dense_layernorm",
        )
        self._key_dense_layer_norm.build(inputs_shape)

        # Value projection
        self._value_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="value",
        )
        self._value_dense.build(inputs_shape)

        # Softmax and dropout
        self._softmax = keras.layers.Softmax(
            axis=-1,
            dtype="float32",
            name="attention_softmax",
        )

        self._dropout_layer = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
        )

        # Output projection
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

        # Multimodal RoPE
        self.multimodal_rotary_embedding = MultimodalRotaryEmbedding(
            mrope_section=self.mrope_section,
            max_wavelength=self.rope_max_wavelength,
            scaling_factor=self.rope_scaling_factor,
            attention_scaling=self.rope_attention_scaling,
            dtype=self.dtype_policy,
        )

        self._dot_product_equation = "bquh,bkuh->buqk"
        self._combine_equation = "buqk,bkuh->bquh"

        self.built = True

    def call(
        self,
        hidden_states,
        position_ids=None,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
        training=None,
    ):
        """Forward pass of multi-head attention with M-RoPE.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_dim).
            position_ids: Position IDs of shape (3, batch, seq_len)
                for M-RoPE, where position_ids[0] = text positions,
                position_ids[1] = temporal positions,
                position_ids[2] = spatial positions. If None, creates
                default sequential positions.
            attention_mask: Attention mask of shape (batch, seq_len, seq_len).
            cache: Optional cached key and value tensors.
            cache_update_index: Index for cache update.
            training: Boolean indicating training mode.

        Returns:
            attention_output: Output tensor after applying attention.
            cache: Updated cache tensors (if cache is provided).
        """
        batch_size = ops.shape(hidden_states)[0]
        seq_len = ops.shape(hidden_states)[1]

        if position_ids is None:
            text_positions = ops.arange(seq_len, dtype="int32")
            text_positions = ops.expand_dims(text_positions, axis=0)
            text_positions = ops.repeat(text_positions, batch_size, axis=0)
            position_ids = ops.stack(
                [text_positions, text_positions, text_positions], axis=0
            )

        # Project to Q, K, V
        query = self._query_dense(hidden_states)
        query = self._query_dense_layer_norm(query)

        def _compute_key_value(x):
            key = self._key_dense(x)
            key = self._key_dense_layer_norm(key)
            value = self._value_dense(x)
            return key, value

        if cache is not None:
            key_cache = cache[:, 0, ...]
            value_cache = cache[:, 1, ...]
            if cache_update_index is None:
                # Use cached keys/values (already have M-RoPE applied)
                key = key_cache
                value = value_cache
                # Still need to apply M-RoPE to query
                query, _ = (
                    self.multimodal_rotary_embedding.apply_multimodal_rotary_embedding(
                        query,
                        key[:, :1, :, :],
                        position_ids,
                    )
                )
            else:
                key_update, value_update = _compute_key_value(hidden_states)

                query, key_update = (
                    self.multimodal_rotary_embedding.apply_multimodal_rotary_embedding(
                        query, key_update, position_ids
                    )
                )

                # Update cache with new key/value
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
            query, key = (
                self.multimodal_rotary_embedding.apply_multimodal_rotary_embedding(
                    query, key, position_ids
                )
            )

        # Repeat K, V for GQA: (batch, seq_len, num_kv_heads, head_dim)
        # -> (batch, seq_len, num_query_heads, head_dim)
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
        attention_output = self._output_dense(attention_output)

        if cache is not None:
            return attention_output, cache
        return attention_output

    def _masked_softmax(self, attention_scores, attention_mask=None):
        """Applies softmax with optional masking.
        Args:
            attention_scores: Attention score tensor.
            attention_mask: Optional mask tensor.

        Returns:
            Masked softmax attention weights.
        """
        if attention_mask is not None:
            return self._softmax(
                attention_scores, attention_mask[:, None, :, :]
            )
        return self._softmax(attention_scores)

    def _compute_attention(
        self, query, key, value, attention_mask=None, cache_update_index=None
    ):
        """Computes attention using query, key, and value tensors.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            attention_mask: Optional mask tensor.
            cache_update_index: Index for sliding window computation.

        Returns:
            attention_output: Output tensor after applying attention.
        """
        # Apply sliding window mask if configured
        if self.sliding_window_size:
            if attention_mask is None:
                query_len = ops.shape(query)[1]
                key_len = ops.shape(key)[1]

                if cache_update_index is not None:
                    causal_mask = ops.arange(key_len) <= (
                        cache_update_index + query_len - 1
                    )
                    causal_mask = ops.cast(causal_mask, dtype="bool")
                    attention_mask = ops.reshape(causal_mask, (1, key_len))
                    attention_mask = ops.broadcast_to(
                        attention_mask, (query_len, key_len)
                    )
                else:
                    attention_mask = ops.tril(
                        ops.ones((query_len, key_len), dtype="bool")
                    )
                attention_mask = ops.expand_dims(attention_mask, 0)
            attention_mask = self._mask_sliding_window(
                attention_mask,
                cache_update_index=cache_update_index
                if cache_update_index is not None
                else 0,
            )

        if fused_attention_op_available():
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

    def _mask_sliding_window(self, attention_mask, cache_update_index=0):
        """Creates and combines a sliding window mask with the attention mask.
        Args:
            attention_mask: Original attention mask.
            cache_update_index: Starting index for the sliding window.

        Returns:
            Combined attention mask with sliding window constraints.
        """
        _, query_len, key_len = ops.shape(attention_mask)
        all_ones = ops.ones((key_len, key_len), "bool")

        if keras.config.backend() == "tensorflow":
            # TODO carried over from qwen3moe
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
                "mrope_section": self.mrope_section,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "rope_attention_scaling": self.rope_attention_scaling,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "dropout": self.dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "sliding_window_size": self.sliding_window_size,
            }
        )
        return config
