import keras


# Removed dependence on einops.
def _rotate_half(x):
    """
    Splits the last dimension of x into two halves and rotates them.

    For an input of shape [..., 2*d], returns a tensor of shape [..., 2*d]
    where the two halves are rotated (i.e. [x1, x2] becomes [-x2, x1]).
    """

    x_shape = keras.ops.shape(x)
    last_dim = x_shape[-1]
    d = last_dim // 2
    x_shape_tensor = keras.ops.convert_to_tensor(x_shape)
    new_shape = keras.ops.concatenate(
        [x_shape_tensor[:-1], keras.ops.convert_to_tensor([d, 2])], axis=0
    )
    x = keras.ops.reshape(x, new_shape)
    x1 = x[..., 0]
    x2 = x[..., 1]
    x_rotated = keras.ops.stack([-x2, x1], axis=-1)
    x_rotated = keras.ops.reshape(x_rotated, x_shape)
    return x_rotated


def _apply_rotary_pos_emb(t, freqs):
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        t: A tensor with shape [..., seq_len, ..., hidden_dim] where the rotary
        embedding is applied to the first `rot_dim` channels of the last
        dimension.
        freqs: A tensor of frequency values with shape [max_seq_len, rot_dim].
        The last `seq_len` entries are used to compute the rotary embeddings.

    Returns:
        A tensor of the same shape as `t` with the rotary positional embeddings
        applied to the first `rot_dim` channels of the last dimension, and the
        remaining channels concatenated unchanged.
    """

    rot_dim = keras.ops.shape(freqs)[-1]
    seq_len = keras.ops.shape(t)[-3]
    orig_dtype = t.dtype
    freqs = freqs[-seq_len:, :]
    freqs = keras.ops.reshape(freqs, (seq_len, 1, rot_dim))
    t_rot = t[..., :rot_dim]
    t_nonrot = t[..., rot_dim:]
    t_rotated = t_rot * keras.ops.cos(freqs) + _rotate_half(
        t_rot
    ) * keras.ops.sin(freqs)
    out = keras.ops.concatenate([t_rotated, t_nonrot], axis=-1)
    return keras.ops.cast(out, orig_dtype)
