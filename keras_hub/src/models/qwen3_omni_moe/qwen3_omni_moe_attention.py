import math

import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.models.qwen3_omni_moe.qwen3_omni_moe_layernorm import Qwen3OmniMoeLayerNorm
from keras_hub.src.utils.keras_utils import clone_initializer


@keras_hub_export("keras_hub.models.Qwen3OmniMoeAttention")
class Qwen3OmniMoeAttention(keras.layers.Layer):
    """Multi-head attention for Qwen3-Omni MoE model.

    This layer implements multi-head attention with grouped query attention (GQA),
    rotary positional embeddings, and Q-Norm/K-Norm for the Qwen3-Omni MoE model.
    It supports efficient key-value caching for autoregressive generation.

    Args:
        num_query_heads: int. The number of heads for the query projections.
        num_key_value_heads: int. The number of heads for the key and value
            projections (must be <= num_query_heads).
        head_dim: int, optional. The size of each attention head.
        rope_max_wavelength: int. Maximum wavelength for RoPE embeddings.
        rope_scaling_factor: float. Scaling factor for RoPE.
        kernel_initializer: Initializer for kernel weights.
        dropout: float, default 0.0. Dropout probability for attention weights.
        layer_norm_epsilon: float, default 1e-6. The epsilon value for layer norm.
        sliding_window_size: int, optional. Size of the sliding local window.

    Example:
    ```python
    # Create attention layer
    attention = Qwen3OmniMoeAttention(
        num_query_heads=32,
        num_key_value_heads=4,
        head_dim=128
    )
    
    # Apply to input
    hidden_states = keras.random.normal((2, 10, 4096))
    outputs = attention(hidden_states)
    ```
    """

    def __init__(
        self,
        num_query_heads,
        num_key_value_heads,
        head_dim=None,
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
        if not self.head_dim:
            self.head_dim = hidden_dim // self.num_query_heads

        self._inv_norm_factor = 1.0 / math.sqrt(self.head_dim)

        # Query projection using EinsumDense for efficient multi-head projection
        self._query_dense = keras.layers.EinsumDense(
            equation="bqm,muh->bquh",
            output_shape=(None, self.num_query_heads, self.head_dim),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="query",
        )
        self._query_dense.build(inputs_shape)

        # Q-Norm: Query normalization (per-head)
        self._query_norm = Qwen3OmniMoeLayerNorm(
            epsilon=self.layer_norm_epsilon,
            head_dim=self.head_dim,
            dtype=self.dtype_policy,
            name="query_norm",
        )
        self._query_norm.build(inputs_shape)

        # Key projection
        self._key_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="key",
        )
        self._key_dense.build(inputs_shape)

        # K-Norm: Key normalization (per-head)
        self._key_norm = Qwen3OmniMoeLayerNorm(
            epsilon=self.layer_norm_epsilon,
            head_dim=self.head_dim,
            dtype=self.dtype_policy,
            name="key_norm",
        )
        self._key_norm.build(inputs_shape)

        # Value projection
        self._value_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="value",
        )
        self._value_dense.build(inputs_shape)

        # Softmax for attention
        self._softmax = keras.layers.Softmax(
            axis=-1,
            dtype="float32",
            name="attention_softmax",
        )

        # Dropout
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

        # Rotary embedding
        self.rotary_embedding_layer = RotaryEmbedding(
            max_wavelength=self.rope_max_wavelength,
            scaling_factor=self.rope_scaling_factor,
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
        """Applies attention mechanism to the input hidden states.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size].
            attention_mask: Mask tensor of shape [batch_size, seq_length, seq_length].
            cache: Optional cached key and value tensors.
            cache_update_index: Index at which to update the cache.
            training: Boolean indicating whether in training mode.

        Returns:
            attention_output: Output tensor after applying attention.
            cache: Updated cache tensors (if cache is provided).
        """
        start_index = cache_update_index if cache_update_index is not None else 0

        # Query projection with Q-Norm
        query = self._query_dense(hidden_states)
        query = self._query_norm(query)

        # Apply RoPE to queries
        query = self.rotary_embedding_layer(query, start_index=start_index)

        def _compute_key_value(x):
            # Key projection with K-Norm
            key = self._key_dense(x)
            key = self._key_norm(key)
            key = self.rotary_embedding_layer(key, start_index=start_index)
            
            # Value projection (no normalization)
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
                    "`cache_update_index` should not be set if `cache` is "
                    f"`None`. Received: cache={cache}, "
                    f"cache_update_index={cache_update_index}"
                )
            key, value = _compute_key_value(hidden_states)

        # Repeat key/value for grouped query attention
        # [batch, seq_len, num_kv_heads, head_dim] -> [batch, seq_len, num_heads, head_dim]
        key = ops.repeat(key, repeats=self.num_key_value_groups, axis=2)
        value = ops.repeat(value, repeats=self.num_key_value_groups, axis=2)

        # Compute attention
        attention_output = self._compute_attention(
            query, key, value, attention_mask
        )

        # Apply dropout
        attention_output = self._dropout_layer(attention_output, training=training)

        # Output projection
        attention_output = self._output_dense(attention_output)

        if cache is not None:
            return attention_output, cache
        return attention_output

    def _masked_softmax(self, attention_scores, attention_mask=None):
        """Applies softmax with optional masking.
        
        Args:
            attention_scores: Tensor of shape [batch, heads, query_len, key_len].
            attention_mask: Optional mask tensor. Can be:
                - 2D: [batch, key_len] - simple padding mask
                - 3D: [batch, query_len, key_len] - causal/attention mask
        """
        if attention_mask is not None:
            # attention_scores shape: [batch, heads, query_len, key_len]
            # We need mask shape: [batch, 1, query_len or 1, key_len]
            mask_ndim = ops.ndim(attention_mask)
            
            if mask_ndim == 2:
                # [batch, key_len] -> [batch, 1, 1, key_len]
                mask = ops.expand_dims(ops.expand_dims(attention_mask, axis=1), axis=1)
            elif mask_ndim == 3:
                # [batch, query_len, key_len] -> [batch, 1, query_len, key_len]
                mask = ops.expand_dims(attention_mask, axis=1)
            else:
                mask = attention_mask
            
            return self._softmax(attention_scores, mask)
        return self._softmax(attention_scores)

    def _compute_attention(self, query, key, value, attention_mask=None):
        """Computes attention using query, key, and value tensors."""
        # Compute attention scores: bquh,bkuh->buqk
        attention_scores = ops.einsum(self._dot_product_equation, query, key)
        
        # Scale by inverse sqrt of head_dim
        attention_scores = ops.multiply(
            attention_scores,
            ops.cast(self._inv_norm_factor, self.compute_dtype),
        )

        # Apply mask and softmax
        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_scores = ops.cast(attention_scores, self.compute_dtype)

        # Apply attention to values: buqk,bkuh->bquh
        attention_output = ops.einsum(self._combine_equation, attention_scores, value)

        return attention_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
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
