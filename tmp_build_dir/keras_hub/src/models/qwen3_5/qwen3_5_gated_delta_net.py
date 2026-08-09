import keras
from keras import ops

from keras_hub.src.models.qwen3_5.qwen3_5_layers import Qwen3_5RMSNormGated


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
    if weight.ndim == 3:
        weight = ops.squeeze(weight, axis=1)
    kernel_size = ops.shape(weight)[-1]
    channels = ops.shape(x)[1]
    x_padded = ops.pad(
        x,
        [[0, 0], [0, 0], [kernel_size - 1, 0]],
    )
    x_cl = ops.transpose(x_padded, (0, 2, 1))
    w = ops.transpose(weight, (1, 0))
    w = ops.expand_dims(w, -1)
    out = ops.depthwise_conv(x_cl, w, strides=1, padding="valid")
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
    padding_mask=None,
):
    """Chunked gated delta rule matching HF's torch_chunk_gated_delta_rule.

    Args:
        query: (B, seq, num_heads, head_k_dim)
        key: (B, seq, num_heads, head_k_dim)
        value: (B, seq, num_heads, head_v_dim)
        g: (B, seq, num_heads) — decay gates (log-space)
        beta: (B, seq, num_heads) — write gates (sigmoid-space)
        chunk_size: Chunk size for blocked computation (default 64).
        initial_state: Optional initial recurrent state.
        output_final_state: Whether to return final state.
        padding_mask: Optional (B, seq) mask.
    Returns:
        output: (B, seq, num_heads, head_v_dim)
        final_state: recurrent state or None
    """
    query = _l2norm(query, axis=-1)
    key = _l2norm(key, axis=-1)

    query = ops.transpose(query, (0, 2, 1, 3))
    key = ops.transpose(key, (0, 2, 1, 3))
    value = ops.transpose(value, (0, 2, 1, 3))
    beta = ops.transpose(beta, (0, 2, 1))
    g = ops.transpose(g, (0, 2, 1))

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

    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad_size > 0:
        query = ops.pad(query, [[0, 0], [0, 0], [0, pad_size], [0, 0]])
        key = ops.pad(key, [[0, 0], [0, 0], [0, pad_size], [0, 0]])
        value = ops.pad(value, [[0, 0], [0, 0], [0, pad_size], [0, 0]])
        beta = ops.pad(beta, [[0, 0], [0, 0], [0, pad_size]])
        g = ops.pad(g, [[0, 0], [0, 0], [0, pad_size]])
    total_len = seq_len + pad_size

    scale = 1.0 / (k_head_dim**0.5)
    query = query * scale

    v_beta = value * ops.expand_dims(beta, -1)
    k_beta = key * ops.expand_dims(beta, -1)

    num_chunks = total_len // chunk_size
    query = ops.reshape(
        query, (batch_size, num_heads, num_chunks, chunk_size, k_head_dim)
    )
    key = ops.reshape(
        key, (batch_size, num_heads, num_chunks, chunk_size, k_head_dim)
    )
    value = ops.reshape(
        value, (batch_size, num_heads, num_chunks, chunk_size, v_head_dim)
    )
    k_beta = ops.reshape(
        k_beta, (batch_size, num_heads, num_chunks, chunk_size, k_head_dim)
    )
    v_beta = ops.reshape(
        v_beta, (batch_size, num_heads, num_chunks, chunk_size, v_head_dim)
    )
    g = ops.reshape(g, (batch_size, num_heads, num_chunks, chunk_size))

    triu_mask = ops.triu(ops.ones((chunk_size, chunk_size)), k=0)
    triu_mask_bool = ops.cast(triu_mask, "bool")

    g = ops.cumsum(g, axis=-1)

    g_row = ops.expand_dims(g, -1)
    g_col = ops.expand_dims(g, -2)
    decay_diff = g_row - g_col
    tril_incl = ops.transpose(
        ops.triu(ops.ones((chunk_size, chunk_size)), k=0), (1, 0)
    )
    decay_diff = decay_diff * tril_incl
    decay_mask = ops.exp(decay_diff) * tril_incl

    kbk = ops.einsum("bhcid,bhcjd->bhcij", k_beta, key)
    attn = -(kbk * decay_mask)
    attn = ops.where(triu_mask_bool, ops.zeros_like(attn), attn)

    # Iterative correction (Neumann-series) matching HF's in-place loop.
    for i in range(1, chunk_size):
        row = attn[..., i : i + 1, :i]
        sub = attn[..., :i, :i]
        correction = ops.matmul(row, sub)
        correction = ops.squeeze(correction, axis=-2)
        new_row = attn[..., i, :]
        update = ops.pad(
            correction, [[0, 0], [0, 0], [0, 0], [0, chunk_size - i]]
        )
        new_row = new_row + update
        before = attn[..., :i, :]
        after = attn[..., i + 1 :, :]
        new_row_exp = ops.expand_dims(new_row, axis=-2)
        attn = ops.concatenate([before, new_row_exp, after], axis=-2)

    attn = attn + ops.eye(chunk_size)

    value = ops.einsum("bhcij,bhcjd->bhcid", attn, v_beta)

    g_exp = ops.exp(g)
    k_cumdecay = ops.einsum(
        "bhcij,bhcjd->bhcid", attn, k_beta * ops.expand_dims(g_exp, -1)
    )

    if initial_state is None:
        last_state = ops.zeros(
            (batch_size, num_heads, k_head_dim, v_head_dim), dtype="float32"
        )
    else:
        last_state = ops.cast(initial_state, "float32")

    triu1_mask = ops.triu(ops.ones((chunk_size, chunk_size)), k=1)
    triu1_bool = ops.cast(triu1_mask, "bool")

    all_chunks_out = []
    for ci in range(num_chunks):
        q_i = query[:, :, ci]
        k_i = key[:, :, ci]
        v_i = value[:, :, ci]
        dm_i = decay_mask[:, :, ci]
        g_i = g[:, :, ci]

        qk = ops.einsum("bhid,bhjd->bhij", q_i, k_i)
        intra_attn = qk * dm_i
        intra_attn = ops.where(
            triu1_bool, ops.zeros_like(intra_attn), intra_attn
        )

        k_cd_i = k_cumdecay[:, :, ci]
        v_prime = ops.einsum("bhid,bhdv->bhiv", k_cd_i, last_state)
        v_new = v_i - v_prime

        attn_inter = ops.einsum(
            "bhid,bhdv->bhiv",
            q_i * ops.expand_dims(ops.exp(g_i), -1),
            last_state,
        )

        chunk_out = attn_inter + ops.einsum(
            "bhij,bhjd->bhid", intra_attn, v_new
        )
        all_chunks_out.append(ops.expand_dims(chunk_out, axis=2))

        g_last = g_i[..., -1]
        state_decay = ops.exp(ops.expand_dims(ops.expand_dims(g_last, -1), -1))
        g_diff = ops.expand_dims(g_last, -1) - g_i
        k_weighted = k_i * ops.expand_dims(ops.exp(g_diff), -1)
        state_update = ops.einsum("bhid,bhiv->bhdv", k_weighted, v_new)
        last_state = last_state * state_decay + state_update

    output = ops.concatenate(all_chunks_out, axis=2)
    output = ops.reshape(output, (batch_size, num_heads, total_len, v_head_dim))
    output = output[:, :, :seq_len, :]

    final_state = last_state if output_final_state else None

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
    padding_mask=None,
):
    """Step-by-step recurrent gated delta rule for inference.

    Matches HF's torch_recurrent_gated_delta_rule. Used for single-step
    autoregressive generation and short sequences.
    """
    query = _l2norm(query, axis=-1)
    key = _l2norm(key, axis=-1)

    query = ops.transpose(query, (0, 2, 1, 3))
    key = ops.transpose(key, (0, 2, 1, 3))
    value = ops.transpose(value, (0, 2, 1, 3))
    beta = ops.transpose(beta, (0, 2, 1))
    g = ops.transpose(g, (0, 2, 1))

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

    if initial_state is None:
        state = ops.zeros(
            (batch_size, num_heads, k_head_dim, v_head_dim), dtype="float32"
        )
    else:
        state = ops.cast(initial_state, "float32")

    all_outputs = []
    for t in range(seq_len):
        q_t = query[:, :, t]
        k_t = key[:, :, t]
        v_t = value[:, :, t]
        g_t = g[:, :, t]
        beta_t = beta[:, :, t]

        g_exp = ops.exp(g_t)
        state = state * ops.expand_dims(ops.expand_dims(g_exp, -1), -1)

        kv_mem = ops.sum(state * ops.expand_dims(k_t, -1), axis=-2)

        delta = (v_t - kv_mem) * ops.expand_dims(beta_t, -1)
        state = state + ops.expand_dims(k_t, -1) * ops.expand_dims(delta, -2)

        out_t = ops.sum(state * ops.expand_dims(q_t, -1), axis=-2)
        all_outputs.append(out_t)

    # (seq_len, batch, heads, v_dim) -> (batch, heads, seq_len, v_dim)
    output = ops.stack(all_outputs, axis=0)
    output = ops.transpose(output, (1, 2, 0, 3))

    final_state = state if output_final_state else None

    output = ops.transpose(output, (0, 2, 1, 3))
    output = ops.cast(output, input_dtype)

    return output, final_state


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
        self.in_proj_qkv = keras.layers.Dense(
            self.key_dim * 2 + self.value_dim,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="in_proj_qkv",
        )
        self.in_proj_qkv.build(input_shape)

        self.in_proj_z = keras.layers.Dense(
            self.value_dim,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="in_proj_z",
        )
        self.in_proj_z.build(input_shape)

        self.in_proj_b = keras.layers.Dense(
            self.num_v_heads,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="in_proj_b",
        )
        self.in_proj_b.build(input_shape)

        self.in_proj_a = keras.layers.Dense(
            self.num_v_heads,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="in_proj_a",
        )
        self.in_proj_a.build(input_shape)

        conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d_weight = self.add_weight(
            name="conv1d_kernel",
            shape=(conv_dim, self.conv_kernel_size),
            initializer="glorot_uniform",
            dtype=self.variable_dtype,
        )

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

        self.norm = Qwen3_5RMSNormGated(
            head_dim=self.head_v_dim,
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="norm",
        )
        self.norm.build((None, self.head_v_dim))

        self.out_proj = keras.layers.Dense(
            self.hidden_size,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="out_proj",
        )
        self.out_proj.build((None, None, self.value_dim))

        self.built = True

    def call(
        self,
        hidden_states,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
        training=None,
    ):
        """Forward pass.

        Args:
            hidden_states: (B, seq_len, hidden_size)
            attention_mask: Optional padding mask.
            cache: Tuple of (conv_state, recurrent_state)
            cache_update_index: Current generation step index.
            training: Whether in training mode.

        Returns:
            output: (B, seq_len, hidden_size)
            cache: (optional) Updated tuple of (conv_state, recurrent_state)
        """
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                mask = ops.cast(
                    ops.expand_dims(attention_mask, -1),
                    hidden_states.dtype,
                )
                hidden_states = hidden_states * mask

        batch_size = ops.shape(hidden_states)[0]
        seq_len = ops.shape(hidden_states)[1]

        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        z = ops.reshape(
            z, (batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        )
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        mixed_qkv_t = ops.transpose(mixed_qkv, (0, 2, 1))

        if cache is not None:
            conv_state, recurrent_state = cache
            if seq_len > 1:
                combined_state = ops.concatenate(
                    [conv_state, mixed_qkv_t], axis=-1
                )

                if attention_mask is not None:
                    valid_lengths = ops.sum(
                        ops.cast(attention_mask, "int32"), axis=-1
                    )
                    indices = ops.expand_dims(
                        valid_lengths + self.conv_kernel_size - 2, axis=-1
                    )
                    offsets = ops.arange(
                        self.conv_kernel_size - 1, dtype="int32"
                    )
                    offsets = offsets - (self.conv_kernel_size - 2)
                    gather_indices = indices + ops.expand_dims(offsets, axis=0)
                    gather_indices = ops.expand_dims(gather_indices, axis=1)
                    gather_indices = ops.repeat(
                        gather_indices, ops.shape(combined_state)[1], axis=1
                    )
                    conv_state = ops.take_along_axis(
                        combined_state, gather_indices, axis=2
                    )
                else:
                    conv_state = combined_state[
                        :, :, -(self.conv_kernel_size - 1) :
                    ]
                padded_input = ops.concatenate(
                    [
                        cache[0][:, :, -(self.conv_kernel_size - 1) :],
                        mixed_qkv_t,
                    ],
                    axis=-1,
                )

                padded_input_transposed = ops.transpose(padded_input, (0, 2, 1))
                conv1d_weight_transposed = ops.transpose(
                    self.conv1d_weight, (1, 0)
                )
                conv1d_weight_expanded = ops.expand_dims(
                    conv1d_weight_transposed, -1
                )

                mixed_qkv_t = ops.depthwise_conv(
                    padded_input_transposed,
                    conv1d_weight_expanded,
                    strides=1,
                    padding="valid",
                )
                mixed_qkv_t = ops.transpose(mixed_qkv_t, (0, 2, 1))

            else:
                sliding_window = ops.concatenate(
                    [conv_state, mixed_qkv_t], axis=-1
                )
                conv_state = sliding_window[:, :, 1:]
                conv1d_weight_expanded = ops.expand_dims(self.conv1d_weight, 0)
                mixed_qkv_t = ops.sum(
                    sliding_window * conv1d_weight_expanded,
                    axis=-1,
                    keepdims=True,
                )

            mixed_qkv_t = mixed_qkv_t * ops.sigmoid(mixed_qkv_t)
            mixed_qkv = ops.transpose(mixed_qkv_t, (0, 2, 1))
        else:
            mixed_qkv_t = _causal_conv1d(mixed_qkv_t, self.conv1d_weight)
            mixed_qkv_t = mixed_qkv_t * ops.sigmoid(mixed_qkv_t)
            mixed_qkv = ops.transpose(mixed_qkv_t, (0, 2, 1))

        query = mixed_qkv[..., : self.key_dim]
        key = mixed_qkv[..., self.key_dim : self.key_dim * 2]
        value = mixed_qkv[..., self.key_dim * 2 :]

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

        beta = ops.sigmoid(b)
        g = -ops.exp(
            ops.cast(self.A_log, "float32")
        ) * keras.activations.softplus(
            ops.cast(a, "float32") + ops.cast(self.dt_bias, "float32")
        )

        if self.num_v_heads // self.num_k_heads > 1:
            repeat_factor = self.num_v_heads // self.num_k_heads
            query = ops.repeat(query, repeats=repeat_factor, axis=2)
            key = ops.repeat(key, repeats=repeat_factor, axis=2)

        if cache is not None:
            if seq_len > 1:
                core_out, last_recurrent_state = _chunk_gated_delta_rule(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=recurrent_state,
                    output_final_state=True,
                    padding_mask=attention_mask,
                )
                cache = (conv_state, last_recurrent_state)
            else:
                core_out, last_recurrent_state = _recurrent_gated_delta_rule(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=recurrent_state,
                    output_final_state=True,
                    padding_mask=attention_mask,
                )
                cache = (conv_state, last_recurrent_state)
        else:
            core_out, _ = _chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                output_final_state=False,
                padding_mask=attention_mask,
            )

        core_out_flat = ops.reshape(core_out, (-1, self.head_v_dim))
        z_flat = ops.reshape(z, (-1, self.head_v_dim))
        core_out_flat = self.norm(core_out_flat)
        core_out_flat = core_out_flat * keras.activations.silu(z_flat)

        core_out = ops.reshape(core_out_flat, (batch_size, seq_len, -1))
        output = self.out_proj(core_out)

        if cache is not None:
            return output, cache
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
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
            }
        )
        return config
