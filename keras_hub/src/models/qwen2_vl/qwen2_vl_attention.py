"""Qwen2-VL Attention layer with Multimodal RoPE (M-RoPE).

This module implements multi-head attention with support for the novel
Multimodal Rotary Position Embedding (M-RoPE), which decomposes RoPE
into temporal, height, and width components for unified text/image/video
position encoding.
"""

import math

import keras
from keras import ops

from keras_hub.src.utils.keras_utils import clone_initializer
from keras_hub.src.utils.keras_utils import fused_attention_op_available


def _rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ops.concatenate((-x2, x1), axis=-1)


def _apply_multimodal_rotary_pos_emb(
    q, k, cos, sin, mrope_section
):
    """Applies M-RoPE to query and key tensors.

    Splits the head dimension into temporal, height, and width sections.
    Each section receives its corresponding positional embedding component.

    Args:
        q: Query tensor of shape `(batch, num_heads, seq_len, head_dim)`.
        k: Key tensor of shape `(batch, num_heads, seq_len, head_dim)`.
        cos: Cosine embeddings of shape `(3, batch, seq_len, head_dim)`.
        sin: Sine embeddings of shape `(3, batch, seq_len, head_dim)`.
        mrope_section: List of 3 ints specifying how many dims for
            each of [temporal, height, width].

    Returns:
        Tuple of rotated query and key tensors.
    """
    # mrope_section is [t_section, h_section, w_section] in terms of
    # half-head-dim. Double it since cos/sin are full head_dim.
    mrope_section_doubled = [s * 2 for s in mrope_section]

    # Split cos and sin along head_dim into sections
    cos_sections = ops.split(
        cos, _cumsum_sections(mrope_section_doubled), axis=-1
    )
    sin_sections = ops.split(
        sin, _cumsum_sections(mrope_section_doubled), axis=-1
    )

    # Pick the right component (temporal=0, height=1, width=2) for each
    # section, cycling through the 3 components.
    cos_parts = []
    sin_parts = []
    for i, (c, s) in enumerate(zip(cos_sections, sin_sections)):
        component = i % 3  # 0=temporal, 1=height, 2=width
        cos_parts.append(c[component])  # (batch, seq_len, section_dim)
        sin_parts.append(s[component])

    cos_combined = ops.expand_dims(
        ops.concatenate(cos_parts, axis=-1), axis=1
    )  # (batch, 1, seq_len, head_dim)
    sin_combined = ops.expand_dims(
        ops.concatenate(sin_parts, axis=-1), axis=1
    )  # (batch, 1, seq_len, head_dim)

    q_embed = q * cos_combined + _rotate_half(q) * sin_combined
    k_embed = k * cos_combined + _rotate_half(k) * sin_combined
    return q_embed, k_embed


def _cumsum_sections(sizes):
    """Convert section sizes to split indices (cumulative sum minus last).

    E.g., [8, 8, 8] -> [8, 16] for use with ops.split.
    """
    result = []
    acc = 0
    for s in sizes[:-1]:
        acc += s
        result.append(acc)
    return result


class Qwen2VLAttention(keras.layers.Layer):
    """Multi-head attention with Multimodal RoPE for Qwen2-VL.

    Supports Grouped-Query Attention (GQA) and sliding window attention.
    Uses separate Q, K, V projections (all with bias) and an output
    projection (without bias).

    The key difference from standard QwenAttention is the M-RoPE:
    position embeddings are provided as `(cos, sin)` of shape
    `(3, batch, seq_len, head_dim)` — one component each for temporal,
    height, and width — combined via `mrope_section`.

    Args:
        num_query_heads: int. Number of query attention heads.
        num_key_value_heads: int. Number of key/value heads (for GQA).
        hidden_dim: int. Model hidden dimension.
        mrope_section: List of 3 ints specifying how many half-head-dim
            elements are allocated to [temporal, height, width].
        rope_max_wavelength: float. Max wavelength for RoPE base.
        kernel_initializer: Initializer for the kernel weights.
        bias_initializer: Initializer for the bias weights.
        dropout: float. Dropout rate for attention weights.
        use_sliding_window_attention: bool. Whether to use sliding window.
        sliding_window_size: int. Sliding window size.
        dtype: string or `keras.mixed_precision.DTypePolicy`.
    """

    def __init__(
        self,
        num_query_heads,
        num_key_value_heads,
        hidden_dim,
        mrope_section,
        rope_max_wavelength=10000,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        dropout=0,
        use_sliding_window_attention=False,
        sliding_window_size=4096,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_dim = hidden_dim
        self.mrope_section = mrope_section
        self.rope_max_wavelength = rope_max_wavelength
        self.dropout = dropout
        self.use_sliding_window_attention = use_sliding_window_attention
        self.sliding_window_size = sliding_window_size

        self.num_key_value_groups = num_query_heads // num_key_value_heads
        self.head_dim = hidden_dim // num_query_heads

        self.kernel_initializer = keras.initializers.get(
            clone_initializer(kernel_initializer)
        )
        self.bias_initializer = keras.initializers.get(
            clone_initializer(bias_initializer)
        )

        self._inv_norm_factor = 1.0 / math.sqrt(self.head_dim)

        # Q, K, V with bias; O without bias
        self._query_dense = keras.layers.EinsumDense(
            equation="bqm,muh->bquh",
            output_shape=(None, self.num_query_heads, self.head_dim),
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            bias_axes="uh",
            dtype=self.dtype_policy,
            name="query",
        )
        self._key_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            bias_axes="vh",
            dtype=self.dtype_policy,
            name="key",
        )
        self._value_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            bias_axes="vh",
            dtype=self.dtype_policy,
            name="value",
        )
        self._output_dense = keras.layers.EinsumDense(
            equation="bquh,uhm->bqm",
            output_shape=(None, self.hidden_dim),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="attention_output",
        )

        self._softmax = keras.layers.Softmax(
            axis=-1,
            dtype="float32",
            name="attention_softmax",
        )
        self._dropout_layer = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
        )

        self._dot_product_equation = "bquh,bkuh->buqk"
        self._combine_equation = "buqk,bkuh->bquh"

    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_embeddings=None,
        cache=None,
        cache_update_index=None,
        training=None,
    ):
        """Forward pass with M-RoPE attention.

        Args:
            hidden_states: Tensor of shape `(batch, seq_len, hidden_dim)`.
            attention_mask: Optional mask of shape
                `(batch, seq_len, seq_len)`.
            position_embeddings: Tuple of `(cos, sin)`, each of shape
                `(3, batch, seq_len, head_dim)`.
            cache: Optional cached key/value states.
            cache_update_index: Index for cache update.
            training: Boolean training mode flag.

        Returns:
            attention_output: Tensor of shape
                `(batch, seq_len, hidden_dim)`.
            cache: Updated cache (if cache was provided).
        """
        query = self._query_dense(hidden_states)

        def _compute_key_value(x):
            key, value = self._key_dense(x), self._value_dense(x)
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
            key, value = _compute_key_value(hidden_states)

        # Apply M-RoPE
        if position_embeddings is not None:
            cos, sin = position_embeddings
            # query: (batch, seq_len, num_heads, head_dim)
            # -> (batch, num_heads, seq_len, head_dim) for RoPE
            query_t = ops.transpose(query, (0, 2, 1, 3))
            key_t = ops.transpose(key, (0, 2, 1, 3))

            query_t, key_t = _apply_multimodal_rotary_pos_emb(
                query_t, key_t, cos, sin, self.mrope_section
            )

            query = ops.transpose(query_t, (0, 2, 1, 3))
            key = ops.transpose(key_t, (0, 2, 1, 3))

        # GQA: repeat key/value heads
        key = ops.repeat(key, repeats=self.num_key_value_groups, axis=2)
        value = ops.repeat(
            value, repeats=self.num_key_value_groups, axis=2
        )

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
            attention_output = ops.dot_product_attention(
                query,
                key,
                value,
                mask=attention_mask,
                scale=self._inv_norm_factor,
            )
            return attention_output

        attention_scores = ops.einsum(
            self._dot_product_equation, query, key
        )
        attention_scores = ops.multiply(
            attention_scores,
            ops.cast(self._inv_norm_factor, self.compute_dtype),
        )
        if self.use_sliding_window_attention:
            attention_mask = self._mask_sliding_window(
                attention_mask,
                cache_update_index=(
                    cache_update_index
                    if cache_update_index is not None
                    else 0
                ),
            )
        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )
        attention_scores = ops.cast(attention_scores, self.compute_dtype)
        attention_output = ops.einsum(
            self._combine_equation, attention_scores, value
        )
        return attention_output

    def _mask_sliding_window(
        self,
        attention_mask,
        cache_update_index=0,
    ):
        _, query_len, key_len = ops.shape(attention_mask)
        all_ones = ops.ones((key_len, key_len), "bool")
        if keras.config.backend() == "tensorflow":
            import tensorflow as tf

            band_size = ops.minimum(
                key_len, self.sliding_window_size - 1
            )
            band_size = ops.cast(band_size, "int32")
            sliding_mask = tf.linalg.band_part(
                all_ones, band_size, band_size
            )
        else:
            sliding_mask = ops.triu(
                all_ones, -1 * self.sliding_window_size + 1
            ) * ops.tril(all_ones, self.sliding_window_size - 1)
        start = (cache_update_index, 0)
        sliding_mask = ops.slice(sliding_mask, start, (query_len, key_len))
        sliding_mask = ops.expand_dims(sliding_mask, 0)
        return ops.logical_and(
            attention_mask, ops.cast(sliding_mask, "bool")
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "hidden_dim": self.hidden_dim,
                "mrope_section": self.mrope_section,
                "rope_max_wavelength": self.rope_max_wavelength,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
                "dropout": self.dropout,
                "use_sliding_window_attention": (
                    self.use_sliding_window_attention
                ),
                "sliding_window_size": self.sliding_window_size,
            }
        )
        return config
