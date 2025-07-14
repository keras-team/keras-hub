from keras import ops

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ops.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, expansion_axis=1):
    cos = ops.expand_dims(cos, expansion_axis)
    sin = ops.expand_dims(sin, expansion_axis)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = ops.shape(hidden_states)
    if n_rep == 1:
        return hidden_states
    hidden_states = ops.expand_dims(hidden_states, axis=2)
    target_shape = (batch, num_key_value_heads, n_rep, slen, head_dim)
    hidden_states = ops.broadcast_to(hidden_states, target_shape)
    return ops.reshape(hidden_states, [batch, num_key_value_heads * n_rep, slen, head_dim])
