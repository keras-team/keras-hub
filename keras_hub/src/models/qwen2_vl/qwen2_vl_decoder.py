"""Qwen2-VL Transformer Decoder layer.

Uses the Qwen2-VL attention with M-RoPE support instead of
standard QwenAttention.
"""

import keras
from keras import ops

from keras_hub.src.models.qwen2_vl.qwen2_vl_attention import (
    Qwen2VLAttention,
)
from keras_hub.src.models.qwen.qwen_layernorm import QwenLayerNorm
from keras_hub.src.utils.keras_utils import clone_initializer


class Qwen2VLTransformerDecoder(keras.layers.Layer):
    """A single Transformer decoder block for Qwen2-VL.

    Structure: RMSNorm → M-RoPE Attention → residual → RMSNorm → SwiGLU
    MLP → residual.

    The key difference from the standard Qwen decoder is that attention
    receives pre-computed M-RoPE position embeddings of shape
    `(3, batch, seq_len, head_dim)`.

    Args:
        intermediate_dim: int. Dimension of the MLP intermediate
            (up/gate) projections.
        hidden_dim: int. Model hidden dimension.
        num_query_heads: int. Number of query heads.
        num_key_value_heads: int. Number of key/value heads.
        mrope_section: list. The M-RoPE section sizes `[t, h, w]`.
        rope_max_wavelength: float. Max wavelength for RoPE base.
        layer_norm_epsilon: float. Epsilon for RMS normalization.
        activation: callable. Activation for the gated MLP.
        kernel_initializer: Initializer for kernels.
        dropout: float. Dropout rate.
        use_sliding_window_attention: bool. Whether to use sliding window.
        sliding_window_size: int. Size of the sliding window.
        dtype: string or `keras.mixed_precision.DTypePolicy`.
    """

    def __init__(
        self,
        intermediate_dim,
        hidden_dim,
        num_query_heads,
        num_key_value_heads,
        mrope_section,
        rope_max_wavelength=10000,
        layer_norm_epsilon=1e-6,
        activation=None,
        kernel_initializer="glorot_uniform",
        dropout=0,
        use_sliding_window_attention=False,
        sliding_window_size=4096,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.hidden_dim = hidden_dim
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.mrope_section = mrope_section
        self.rope_max_wavelength = rope_max_wavelength
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout
        self.use_sliding_window_attention = use_sliding_window_attention
        self.sliding_window_size = sliding_window_size

        if activation is None:
            activation = ops.silu
        self.activation = activation

        self.kernel_initializer = keras.initializers.get(
            clone_initializer(kernel_initializer)
        )

        self._self_attention_layer = Qwen2VLAttention(
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_dim=hidden_dim,
            mrope_section=mrope_section,
            rope_max_wavelength=rope_max_wavelength,
            kernel_initializer=clone_initializer(kernel_initializer),
            dropout=dropout,
            use_sliding_window_attention=use_sliding_window_attention,
            sliding_window_size=sliding_window_size,
            dtype=self.dtype_policy,
            name="self_attention",
        )

        self._self_attention_layernorm = QwenLayerNorm(
            epsilon=layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="self_attention_layernorm",
        )

        self._feedforward_layernorm = QwenLayerNorm(
            epsilon=layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="feedforward_layernorm",
        )

        # SwiGLU MLP: gate_proj and up_proj -> activation -> down_proj
        self._feedforward_gate_dense = keras.layers.Dense(
            intermediate_dim,
            use_bias=False,
            kernel_initializer=clone_initializer(kernel_initializer),
            dtype=self.dtype_policy,
            name="feedforward_gate_dense",
        )
        self._feedforward_intermediate_dense = keras.layers.Dense(
            intermediate_dim,
            use_bias=False,
            kernel_initializer=clone_initializer(kernel_initializer),
            dtype=self.dtype_policy,
            name="feedforward_intermediate_dense",
        )
        self._feedforward_output_dense = keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            kernel_initializer=clone_initializer(kernel_initializer),
            dtype=self.dtype_policy,
            name="feedforward_output_dense",
        )

    def build(self, decoder_sequence_shape):
        self._self_attention_layernorm.build(decoder_sequence_shape)
        self._self_attention_layer.build(decoder_sequence_shape)
        self._feedforward_layernorm.build(decoder_sequence_shape)
        self._feedforward_gate_dense.build(decoder_sequence_shape)
        self._feedforward_intermediate_dense.build(decoder_sequence_shape)
        self._feedforward_output_dense.build(
            decoder_sequence_shape[:-1] + (self.intermediate_dim,)
        )
        self.built = True

    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_embeddings=None,
        cache=None,
        cache_update_index=None,
        training=None,
    ):
        """Forward pass through the decoder block.

        Args:
            hidden_states: Input tensor of shape
                `(batch, seq_len, hidden_dim)`.
            attention_mask: Optional mask of shape
                `(batch, seq_len, seq_len)`.
            position_embeddings: Tuple of `(cos, sin)`, each of shape
                `(3, batch, seq_len, head_dim)` for M-RoPE.
            cache: Optional cached key/value states.
            cache_update_index: Index for cache update.
            training: Boolean training flag.

        Returns:
            hidden_states: Output tensor.
            cache: Updated cache (if provided).
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self._self_attention_layernorm(hidden_states)

        attention_output = self._self_attention_layer(
            hidden_states,
            attention_mask=self._compute_self_attention_mask(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                cache=cache,
                cache_update_index=cache_update_index,
            ),
            position_embeddings=position_embeddings,
            cache=cache,
            cache_update_index=cache_update_index,
            training=training,
        )

        if cache is not None:
            attention_output, cache = attention_output, cache
            if isinstance(attention_output, tuple):
                attention_output, cache = attention_output

        hidden_states = residual + attention_output

        # SwiGLU MLP with residual
        residual = hidden_states
        hidden_states = self._feedforward_layernorm(hidden_states)

        gate = self.activation(
            self._feedforward_gate_dense(hidden_states)
        )
        hidden_states = self._feedforward_intermediate_dense(hidden_states)
        hidden_states = self._feedforward_output_dense(
            gate * hidden_states
        )

        hidden_states = residual + hidden_states

        if cache is not None:
            return hidden_states, cache
        return hidden_states

    def _compute_self_attention_mask(
        self,
        hidden_states,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
    ):
        """Computes the causal self-attention mask.

        Combines padding mask with causal mask. During generation with
        cache, only produces mask for the current token(s).
        """
        batch_size = ops.shape(hidden_states)[0]
        input_length = ops.shape(hidden_states)[1]

        if cache is not None:
            output_length = ops.shape(cache)[2]
        else:
            output_length = input_length

        # Causal mask
        causal_mask = ops.triu(
            ops.ones((output_length, output_length), dtype="bool"),
            k=1,
        )
        causal_mask = ops.logical_not(causal_mask)

        if cache_update_index is not None:
            # Slice for the current step
            causal_mask = ops.slice(
                causal_mask,
                (cache_update_index, 0),
                (input_length, output_length),
            )

        # Combine with padding mask
        if attention_mask is not None:
            attention_mask = ops.expand_dims(attention_mask, axis=1)
            attention_mask = ops.cast(attention_mask, dtype="bool")
            causal_mask = ops.expand_dims(causal_mask, axis=0)
            causal_mask = ops.broadcast_to(
                causal_mask,
                (batch_size, input_length, output_length),
            )
            causal_mask = ops.logical_and(causal_mask, attention_mask)
        else:
            causal_mask = ops.expand_dims(causal_mask, axis=0)
            causal_mask = ops.broadcast_to(
                causal_mask,
                (batch_size, input_length, output_length),
            )

        return causal_mask

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "hidden_dim": self.hidden_dim,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "mrope_section": self.mrope_section,
                "rope_max_wavelength": self.rope_max_wavelength,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "use_sliding_window_attention": (
                    self.use_sliding_window_attention
                ),
                "sliding_window_size": self.sliding_window_size,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
            }
        )
        return config
