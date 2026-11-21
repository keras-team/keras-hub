from keras import ops


def rotate_half(x):
    x1 = x[..., : ops.shape(x)[-1] // 2]
    x2 = x[..., ops.shape(x)[-1] // 2 :]
    return ops.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, expansion_axis=1):
    cos = ops.expand_dims(cos, expansion_axis)
    sin = ops.expand_dims(sin, expansion_axis)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_single(tensor, cos, sin, expansion_axis=1):
    cos = ops.expand_dims(cos, expansion_axis)
    sin = ops.expand_dims(sin, expansion_axis)
    tensor_embed = (tensor * cos) + (rotate_half(tensor) * sin)
    return tensor_embed


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = ops.shape(hidden_states)
    if n_rep == 1:
        return hidden_states
    hidden_states = ops.expand_dims(hidden_states, axis=2)
    target_shape = (batch, num_key_value_heads, n_rep, slen, head_dim)
    hidden_states = ops.broadcast_to(hidden_states, target_shape)
    return ops.reshape(
        hidden_states, [batch, num_key_value_heads * n_rep, slen, head_dim]
    )


def rope_init(rope_theta, partial_rotary_factor, head_dim):
    """Initialize RoPE (Rotary Position Embedding) parameters.

    Args:
        rope_theta: float. The theta value for RoPE.
        partial_rotary_factor: float. The factor for partial rotary embedding.
        head_dim: int. The dimension of each attention head.

    Returns:
        A tuple of (inv_freq, attention_scaling) where inv_freq is the inverse
        frequency tensor and attention_scaling is the scaling factor.
    """
    base = rope_theta
    dim = int(head_dim * partial_rotary_factor)

    inv_freq = 1.0 / (
        ops.power(base, ops.arange(0, dim, 2, dtype="float32") / dim)
    )
    attention_scaling = 1.0
    return inv_freq, attention_scaling
