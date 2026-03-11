"""Gated Delta Net linear attention layer for Qwen3.5.

This implements a recurrent linear attention mechanism that replaces
standard softmax attention in some layers. It uses:
- Causal Conv1d for local context mixing
- Delta rule recurrence for long-range memory
- Gating mechanisms (beta for write, g for decay, z for output)

Reference: HF transformers Qwen3NextGatedDeltaNet / Qwen3_5GatedDeltaNet
"""

import keras
from keras import ops

from keras_hub.src.models.qwen3_5.qwen3_5_layernorm import Qwen3_5LayerNorm


def _l2norm(x, axis=-1, eps=1e-6):
    """L2 normalize along the given axis."""
    inv_norm = ops.rsqrt(ops.sum(x * x, axis=axis, keepdims=True) + eps)
    return x * inv_norm


def _causal_conv1d(x, weight, bias=None):
    """Apply depthwise causal conv1d.

    Args:
        x: (batch, channels, seq_len)
        weight: (channels, 1, kernel_size) or (channels, kernel_size)
        bias: (channels,) or None

    Returns:
        (batch, channels, seq_len)
    """
    if weight.ndim == 2:
        weight = ops.expand_dims(weight, 1)
    kernel_size = ops.shape(weight)[-1]
    channels = ops.shape(x)[1]

    # Left-pad for causal convolution.
    x_padded = ops.pad(
        x,
        [[0, 0], [0, 0], [kernel_size - 1, 0]],
    )

    # Depthwise conv1d: process each channel independently.
    # Reshape for grouped conv: (batch, 1, seq, channels)
    x_padded = ops.transpose(x_padded, (0, 2, 1))
    x_padded = ops.expand_dims(x_padded, 1)

    # Weight shape for conv: (kernel_size, 1, channels)
    # Flip weight for cross-correlation -> convolution.
    w = ops.transpose(weight, (2, 1, 0))
    w = ops.flip(w, axis=0)

    # Use depthwise conv.
    out = ops.depthwise_conv(x_padded, w, strides=1, padding="valid")

    # out shape: (batch, seq_len, 1, channels) -> (batch, channels, seq_len)
    out = ops.squeeze(out, axis=2)
    out = ops.transpose(out, (0, 2, 1))

    if bias is not None:
        out = out + ops.reshape(bias, (1, channels, 1))
    return out


def _chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
):
    """Chunked gated delta rule for training (parallel over chunks).

    Args:
        query: (B, seq, num_heads, head_k_dim)
        key: (B, seq, num_heads, head_k_dim)
        value: (B, seq, num_heads, head_v_dim)
        g: (B, seq, num_heads) — decay gates (log-space)
        beta: (B, seq, num_heads) — write gates (sigmoid-space)
        chunk_size: Chunk size for blocked computation.
        initial_state: Optional initial recurrent state.
        output_final_state: Whether to return final state.

    Returns:
        output: (B, seq, num_heads, head_v_dim)
        final_state: recurrent state or None
    """
    # L2-normalize Q and K.
    query = _l2norm(query, axis=-1)
    key = _l2norm(key, axis=-1)

    # Transpose to (B, heads, seq, dim).
    query = ops.transpose(query, (0, 2, 1, 3))
    key = ops.transpose(key, (0, 2, 1, 3))
    value = ops.transpose(value, (0, 2, 1, 3))
    beta = ops.transpose(beta, (0, 2, 1))
    g = ops.transpose(g, (0, 2, 1))

    # Cast to float32 for numerical stability.
    input_dtype = query.dtype
    query = ops.cast(query, "float32")
    key = ops.cast(key, "float32")
    value = ops.cast(value, "float32")
    beta = ops.cast(beta, "float32")
    g = ops.cast(g, "float32")

    batch_size = ops.shape(key)[0]
    num_heads = ops.shape(key)[1]
    seq_len = ops.shape(key)[2]
    k_head_dim = ops.shape(key)[3]
    v_head_dim = ops.shape(value)[3]

    scale = 1.0 / (k_head_dim**0.5)
    query = query * scale

    v_beta = value * ops.expand_dims(beta, -1)

    # Initialize recurrent state.
    if initial_state is None:
        state = ops.zeros(
            (batch_size, num_heads, k_head_dim, v_head_dim),
            dtype="float32",
        )
    else:
        state = ops.cast(initial_state, "float32")

    # Process chunks using a simple loop.
    # For simplicity, we process the entire sequence in one pass
    # using the recurrent formulation (equivalent to chunked but
    # without the chunked optimization for now).

    outputs = []
    for t in range(seq_len):
        q_t = query[:, :, t, :]
        k_t = key[:, :, t, :]
        v_beta_t = v_beta[:, :, t, :]
        g_t = g[:, :, t]

        # Decay the state.
        decay = ops.exp(ops.expand_dims(ops.expand_dims(g_t, -1), -1))
        state = state * decay

        # Delta update: compute what the current state predicts
        # for v given k, then add correction.
        kv_pred = ops.sum(state * ops.expand_dims(k_t, -1), axis=-2)
        delta = v_beta_t - kv_pred * ops.expand_dims(beta[:, :, t], -1)
        state = state + ops.expand_dims(k_t, -1) * ops.expand_dims(delta, -2)

        # Query the state.
        out_t = ops.sum(state * ops.expand_dims(q_t, -1), axis=-2)
        outputs.append(out_t)

    output = ops.stack(outputs, axis=2)

    final_state = state if output_final_state else None

    # Transpose back to (B, seq, heads, v_dim).
    output = ops.transpose(output, (0, 2, 1, 3))
    output = ops.cast(output, input_dtype)

    return output, final_state


def _recurrent_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    initial_state=None,
    output_final_state=False,
):
    """Step-by-step recurrent gated delta rule for inference.

    Same signature as _chunk_gated_delta_rule but processes one step
    at a time (optimized for autoregressive generation).
    """
    return _chunk_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        chunk_size=1,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )


class Qwen3_5GatedDeltaNet(keras.layers.Layer):
    """Gated Delta Net linear attention for Qwen3.5.

    Replaces standard self-attention in ``linear_attention`` layers.
    Uses a delta rule recurrence with gating for efficient
    sequence modeling.

    Args:
        hidden_size: Model hidden dimension.
        linear_num_key_heads: Number of key heads for linear attention.
        linear_num_value_heads: Number of value heads for linear
            attention.
        linear_key_head_dim: Dimension per key head.
        linear_value_head_dim: Dimension per value head.
        linear_conv_kernel_dim: Kernel size for causal conv1d.
        hidden_activation: Activation function name.
        layer_norm_epsilon: Epsilon for RMSNorm.
        kernel_initializer: Initializer for dense layers.
    """

    def __init__(
        self,
        hidden_size,
        linear_num_key_heads,
        linear_num_value_heads,
        linear_key_head_dim,
        linear_value_head_dim,
        linear_conv_kernel_dim=4,
        hidden_activation="silu",
        layer_norm_epsilon=1e-6,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_k_heads = linear_num_key_heads
        self.num_v_heads = linear_num_value_heads
        self.head_k_dim = linear_key_head_dim
        self.head_v_dim = linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = linear_conv_kernel_dim
        self.hidden_activation = hidden_activation
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

    def build(self, input_shape):
        # Qwen3.5 has separate projections for QKV, Z, B, A.
        # QKV fused projection.
        self.in_proj_qkv = keras.layers.Dense(
            self.key_dim * 2 + self.value_dim,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="in_proj_qkv",
        )
        self.in_proj_qkv.build(input_shape)

        # Z (output gate) projection.
        self.in_proj_z = keras.layers.Dense(
            self.value_dim,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="in_proj_z",
        )
        self.in_proj_z.build(input_shape)

        # Beta (write gate) projection.
        self.in_proj_b = keras.layers.Dense(
            self.num_v_heads,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="in_proj_b",
        )
        self.in_proj_b.build(input_shape)

        # A (decay gate) projection.
        self.in_proj_a = keras.layers.Dense(
            self.num_v_heads,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="in_proj_a",
        )
        self.in_proj_a.build(input_shape)

        # Causal conv1d (depthwise).
        conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d_weight = self.add_weight(
            name="conv1d/kernel",
            shape=(conv_dim, self.conv_kernel_size),
            initializer="glorot_uniform",
            dtype=self.variable_dtype,
        )
        self.conv1d_bias = self.add_weight(
            name="conv1d/bias",
            shape=(conv_dim,),
            initializer="zeros",
            dtype=self.variable_dtype,
        )

        # dt_bias and A_log (learnable parameters for decay).
        self.dt_bias = self.add_weight(
            name="dt_bias",
            shape=(self.num_v_heads,),
            initializer="ones",
            dtype=self.variable_dtype,
        )
        self.A_log = self.add_weight(
            name="A_log",
            shape=(self.num_v_heads,),
            initializer="zeros",
            dtype=self.variable_dtype,
        )

        # Output gated RMSNorm.
        self.norm = Qwen3_5LayerNorm(
            head_dim=self.head_v_dim,
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="norm",
        )
        self.norm.build((None, self.head_v_dim))

        # Output projection.
        self.out_proj = keras.layers.Dense(
            self.hidden_size,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="out_proj",
        )
        self.out_proj.build((None, None, self.value_dim))

        self.built = True

    def call(self, hidden_states, attention_mask=None, training=None):
        """Forward pass.

        Args:
            hidden_states: (B, seq_len, hidden_size)
            attention_mask: Optional padding mask.
            training: Whether in training mode.

        Returns:
            output: (B, seq_len, hidden_size)
        """
        # Mask padding states.
        if attention_mask is not None:
            # attention_mask: (B, seq_len) with 1 for valid, 0 for pad.
            if attention_mask.ndim == 2:
                mask = ops.cast(
                    ops.expand_dims(attention_mask, -1),
                    hidden_states.dtype,
                )
                hidden_states = hidden_states * mask

        batch_size = ops.shape(hidden_states)[0]
        seq_len = ops.shape(hidden_states)[1]

        # Project QKV.
        mixed_qkv = self.in_proj_qkv(hidden_states)

        # Project gating signals.
        z = self.in_proj_z(hidden_states)
        z = ops.reshape(
            z, (batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        )

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # Causal conv1d on QKV.
        # Transpose to (B, channels, seq_len) for conv.
        mixed_qkv_t = ops.transpose(mixed_qkv, (0, 2, 1))
        mixed_qkv_t = _causal_conv1d(
            mixed_qkv_t, self.conv1d_weight, self.conv1d_bias
        )
        # Apply SiLU activation after conv.
        mixed_qkv_t = mixed_qkv_t * ops.sigmoid(mixed_qkv_t)
        mixed_qkv = ops.transpose(mixed_qkv_t, (0, 2, 1))

        # Split QKV.
        query, key, value = ops.split(
            mixed_qkv,
            [self.key_dim, self.key_dim * 2],
            axis=-1,
        )

        query = ops.reshape(
            query,
            (batch_size, seq_len, self.num_k_heads, self.head_k_dim),
        )
        key = ops.reshape(
            key,
            (batch_size, seq_len, self.num_k_heads, self.head_k_dim),
        )
        value = ops.reshape(
            value,
            (batch_size, seq_len, self.num_v_heads, self.head_v_dim),
        )

        # Compute decay gate.
        beta = ops.sigmoid(b)
        g = -ops.exp(
            ops.cast(self.A_log, "float32")
        ) * keras.activations.softplus(
            ops.cast(a, "float32") + ops.cast(self.dt_bias, "float32")
        )

        # Expand K heads to match V heads if needed.
        if self.num_v_heads // self.num_k_heads > 1:
            repeat_factor = self.num_v_heads // self.num_k_heads
            query = ops.repeat(query, repeats=repeat_factor, axis=2)
            key = ops.repeat(key, repeats=repeat_factor, axis=2)

        # Apply gated delta rule.
        core_out, _ = _chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            output_final_state=False,
        )

        # Output gated normalization.
        # Reshape to (B * seq * heads, v_dim) for norm.
        core_out_flat = ops.reshape(core_out, (-1, self.head_v_dim))
        z_flat = ops.reshape(z, (-1, self.head_v_dim))
        core_out_flat = self.norm(core_out_flat)
        core_out_flat = core_out_flat * ops.sigmoid(z_flat)

        # Reshape back and project.
        core_out = ops.reshape(core_out_flat, (batch_size, seq_len, -1))
        output = self.out_proj(core_out)
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "linear_num_key_heads": self.num_k_heads,
                "linear_num_value_heads": self.num_v_heads,
                "linear_key_head_dim": self.head_k_dim,
                "linear_value_head_dim": self.head_v_dim,
                "linear_conv_kernel_dim": self.conv_kernel_size,
                "hidden_activation": self.hidden_activation,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config
