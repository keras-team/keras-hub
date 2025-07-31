from keras import layers
from keras import ops
from keras import random


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


def eager_attention_forward(
    module,
    query,
    key,
    value,
    scaling,
    attention_mask=None,
    dropout=0.0,
    training=False,
):
    softmax_op = layers.Softmax(
        axis=-1,
        dtype="float32",
        name="attention_softmax",
    )

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = (
        ops.matmul(query, ops.transpose(key_states, axes=(0, 1, 3, 2)))
        * scaling
    )

    if attention_mask is not None:
        attn_weights = softmax_op(attn_weights, attention_mask[:, None, :, :])
    else:
        attn_weights = softmax_op(attn_weights)

    if training:
        attn_weights = random.dropout(attn_weights, rate=dropout)
    attn_output = ops.matmul(attn_weights, value_states)
    attn_output = ops.transpose(attn_output, axes=(0, 2, 1, 3))

    return attn_output


def rope_init(rope_theta: float, partial_rotary_factor: float, head_dim: int):
    base = rope_theta
    dim = int(head_dim * partial_rotary_factor)

    inv_freq = 1.0 / (
        ops.power(base, ops.arange(0, dim, 2, dtype="float32") / dim)
    )
    attention_scaling = 1.0
    return inv_freq, attention_scaling
