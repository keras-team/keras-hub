import keras


def rotate_half(x):
    """Rotates half of the hidden dimensions of the input tensor.

    This function is used to implement rotary positional embeddings. It splits
    the last dimension of the input tensor into two halves, negates the second
    half, and then concatenates them back together.

    Args:
        x: The input tensor.

    Returns:
        A new tensor with the second half of the last dimension rotated.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return keras.ops.concatenate([-x2, x1], axis=-1)


def repeat_kv(hidden_states, n_rep):
    """Repeats the key and value states for Grouped-Query Attention.

    This function is used in Grouped-Query Attention (GQA) to expand the key
    and value states to match the number of query heads.

    Args:
        hidden_states: The key or value tensor to be repeated, with a shape of
            `[batch, num_key_value_heads, seq_len, head_dim]`.
        n_rep: int. The number of times to repeat the key/value heads.

    Returns:
        The repeated tensor with a shape of
        `[batch, num_key_value_heads * n_rep, seq_len, head_dim]`.
    """
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = keras.ops.expand_dims(hidden_states, 2)
    hidden_states = keras.ops.repeat(hidden_states, n_rep, axis=2)
    return keras.ops.reshape(
        hidden_states, (batch, num_key_value_heads * n_rep, slen, head_dim)
    )


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    """Applies rotary positional embedding to the input tensor.

    Args:
        x: The input tensor.
        cos: The cosine part of the rotary embedding.
        sin: The sine part of the rotary embedding.
        unsqueeze_dim: int. The dimension to unsqueeze `cos` and `sin` before
            applying the embedding. Defaults to 1.

    Returns:
        The tensor with rotary positional embeddings applied.
    """
    cos = keras.ops.expand_dims(cos, axis=unsqueeze_dim)
    sin = keras.ops.expand_dims(sin, axis=unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def eager_attention_forward(
    query,
    key,
    value,
    num_key_value_groups,
    head_dim,
    attention_mask,
    dropout=0.0,
    scaling=None,
    softcap=None,
    training=False,
):
    """Forward pass for an eager attention implementation.

    Args:
        query: The query tensor.
        key: The key tensor.
        value: The value tensor.
        num_key_value_groups: int. The number of key-value groups.
        head_dim: int. The dimension of each attention head.
        attention_mask: The attention mask to apply.
        dropout: float. The dropout rate. Defaults to 0.0.
        scaling: float, optional. The scaling factor for attention scores.
            If `None`, it defaults to `head_dim**-0.5`.
        softcap: float, optional. A softcap value to apply to attention weights.
            Defaults to `None`.
        training: bool. Whether the model is in training mode. Defaults to
            `False`.
    """
    if scaling is None:
        scaling = head_dim**-0.5
    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)
    attn_weights = (
        keras.ops.matmul(query, keras.ops.transpose(key_states, (0, 1, 3, 2)))
        * scaling
    )
    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = keras.ops.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + keras.ops.cast(
            causal_mask, dtype=attn_weights.dtype
        )
    attn_weights_dtype = attn_weights.dtype
    attn_weights = keras.ops.softmax(
        keras.ops.cast(attn_weights, "float32"), axis=-1
    )
    attn_weights = keras.ops.cast(attn_weights, attn_weights_dtype)
    if training:
        attn_weights = keras.layers.Dropout(dropout)(
            attn_weights, training=training
        )
    attn_output = keras.ops.matmul(attn_weights, value_states)
    attn_output = keras.ops.transpose(attn_output, (0, 2, 1, 3))
    return attn_output, attn_weights
