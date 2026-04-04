import math

import keras
from keras import ops

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.models.qwen3_5.qwen3_5_layernorm import Qwen3_5LayerNorm
from keras_hub.src.utils.keras_utils import clone_initializer
from keras_hub.src.utils.keras_utils import fused_attention_op_available


def _apply_mrope(
    x_rope, rotary_embedding_layer, position_ids, mrope_section, rotary_dim
):
    """Apply interleaved Multi-dimensional RoPE (M-RoPE).

    For multimodal inputs, position_ids has shape (batch, 4, seq_len) where
    the 4 channels are [text, temporal, height, width]. Each channel covers a
    different slice of the rotary dimensions based on mrope_section.

    Args:
        x_rope: Tensor (batch, seq, heads, rotary_dim) — the portion of Q or K
            that receives RoPE.
        rotary_embedding_layer: RotaryEmbedding instance.
        position_ids: int32 tensor (batch, 4, seq_len) or None.
            When None, standard sequential positions are used.
        mrope_section: list[int] — [s_t, s_h, s_w] sizes (in pairs of dims)
            for each of the 3 spatial channels. The text channel mirrors
            the first spatial channel's positions.
        rotary_dim: int — total rotary dimension.
    Returns:
        Tensor (batch, seq, heads, rotary_dim).
    """
    if position_ids is None:
        # Plain 1D RoPE: let RotaryEmbedding handle sequentially.
        return rotary_embedding_layer(x_rope, start_index=0)

    # position_ids: (batch, 4, seq_len)
    # channels: 0=text, 1=temporal, 2=height, 3=width
    # mrope_section: [s_t, s_h, s_w] — number of
    #  *pairs* of dims per channel
    # Total pairs = rotary_dim // 2
    s_t, s_h = mrope_section[0], mrope_section[1]

    # We compute full cos/sin for each of the 3 spatial channels, then pick
    # the appropriate slices based on mrope_section.  Channel 0 (text) uses
    # the temporal (ch1) position ids — they coincide for text tokens.
    device_dtype = x_rope.dtype

    def _get_cos_sin(pos_ids):
        """pos_ids: (batch, seq_len) → cos/sin: (batch, seq, rotary_dim)."""
        # Use RotaryEmbedding._compute_cos_sin_embedding via the `positions`
        # argument so we bypass the sequential position logic.
        dummy = x_rope[:, :, 0, :]  # (batch, seq, rotary_dim)
        dummy_moved = ops.moveaxis(dummy, -1, -1)  # noop for shape propagation
        cos_emb, sin_emb = rotary_embedding_layer._compute_cos_sin_embedding(
            dummy_moved, start_index=0, positions=ops.cast(pos_ids, "float32")
        )
        return cos_emb, sin_emb  # (batch, seq, rotary_dim)

    # Build a combined cos/sin by interleaving sections.
    # Section layout (pairs of dims):
    #   [0 : s_t]           → temporal channel (ch 1)
    #   [s_t : s_t+s_h]     → height  channel (ch 2)
    #   [s_t+s_h : total]   → width   channel (ch 3)
    # For text tokens all channels hold the same sequential position id,
    # so the specific assignment doesn't matter.
    cos_t, sin_t = _get_cos_sin(position_ids[:, 1, :])  # temporal
    cos_h, sin_h = _get_cos_sin(position_ids[:, 2, :])  # height
    cos_w, sin_w = _get_cos_sin(position_ids[:, 3, :])  # width

    # Slice each embedding to its own section (pairs of dims → *2 for actual).
    t_end = s_t * 2
    h_end = t_end + s_h * 2
    # w covers the remainder

    cos_emb = ops.concatenate(
        [cos_t[..., :t_end], cos_h[..., t_end:h_end], cos_w[..., h_end:]],
        axis=-1,
    )  # (batch, seq, rotary_dim)
    sin_emb = ops.concatenate(
        [sin_t[..., :t_end], sin_h[..., t_end:h_end], sin_w[..., h_end:]],
        axis=-1,
    )  # (batch, seq, rotary_dim)

    # Apply: x_rope is (batch, seq, heads, rotary_dim)
    # Expand embeddings for heads dim.
    cos_emb = ops.expand_dims(cos_emb, axis=2)  # (B, seq, 1, rot)
    sin_emb = ops.expand_dims(sin_emb, axis=2)

    x1, x2 = x_rope[..., : rotary_dim // 2], x_rope[..., rotary_dim // 2 :]
    rotated = ops.concatenate([-x2, x1], axis=-1)
    return ops.cast(
        (ops.cast(x_rope, "float32") * ops.cast(cos_emb, "float32"))
        + (ops.cast(rotated, "float32") * ops.cast(sin_emb, "float32")),
        device_dtype,
    )


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
        mrope_section=None,
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
        # mrope_section: [s_t, s_h, s_w] in number of *pairs* of rotary dims.
        # When None, standard 1D RoPE is used (text-only mode).
        self.mrope_section = mrope_section
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

    def _apply_partial_rope(self, x, start_index, position_ids=None):
        """Apply RoPE only to the first `rotary_dim` dimensions.

        When `position_ids` is a 4-channel tensor (M-RoPE), delegates to the
        `_apply_mrope` function which handles each spatial channel separately.
        When None, falls back to sequential standard RoPE.
        """
        if self.mrope_section is not None and position_ids is not None:
            # Multimodal path: M-RoPE with 4-channel position IDs.
            x_rope = x[..., : self.rotary_dim]
            x_pass = x[..., self.rotary_dim :]
            x_rope = _apply_mrope(
                x_rope,
                self.rotary_embedding_layer,
                position_ids,
                self.mrope_section,
                self.rotary_dim,
            )
            if self.rotary_dim == self.head_dim:
                return x_rope
            return ops.concatenate([x_rope, x_pass], axis=-1)
        else:
            # Standard 1D RoPE path.
            if self.rotary_dim == self.head_dim:
                return self.rotary_embedding_layer(x, start_index=start_index)
            x_rope = x[..., : self.rotary_dim]
            x_pass = x[..., self.rotary_dim :]
            x_rope = self.rotary_embedding_layer(
                x_rope, start_index=start_index
            )
            return ops.concatenate([x_rope, x_pass], axis=-1)

    def call(
        self,
        hidden_states,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
        position_ids=None,
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
        query = self._apply_partial_rope(query, start_index, position_ids)

        def _compute_key_value(x):
            key = self._key_dense(x)
            key = self._key_norm(key)
            key = self._apply_partial_rope(key, start_index, position_ids)
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
                "mrope_section": self.mrope_section,
            }
        )
        return config
