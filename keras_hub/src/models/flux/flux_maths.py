import keras
from keras import ops


class TimestepEmbedding(keras.layers.Layer):
    """Creates sinusoidal timestep embeddings.

    Call arguments:
        t: Tensor of shape (N,), representing N indices, one per batch element.
            These values may be fractional.
        dim: int. The dimension of the output.
        max_period: int, optional. Controls the minimum frequency of the
            embeddings. Defaults to 10000.
        time_factor: float, optional. A scaling factor applied to `t`. Defaults
            to 1000.0.

    Returns:
        A tensor of shape (N, D) representing the positional embeddings,
        where N is the number of batch elements and D is the specified
        dimension `dim`.
    """

    def call(self, t, dim, max_period=10000, time_factor=1000.0):
        t = time_factor * t
        half_dim = dim // 2
        freqs = ops.exp(
            ops.cast(-ops.log(max_period), dtype=t.dtype)
            * ops.arange(half_dim, dtype=t.dtype)
            / half_dim
        )
        args = t[:, None] * freqs[None]
        embedding = ops.concatenate([ops.cos(args), ops.sin(args)], axis=-1)

        if dim % 2 != 0:
            embedding = ops.concatenate(
                [embedding, ops.zeros_like(embedding[:, :1])], axis=-1
            )

        return embedding


class RotaryPositionalEmbedding(keras.layers.Layer):
    """
    Applies Rotary Positional Embedding (RoPE) to the input tensor.

    Call arguments:
        pos: KerasTensor. The positional tensor with shape (..., n, d).
        dim: int. The embedding dimension, should be even.
        theta: int. The base frequency.

    Returns:
        KerasTensor: The tensor with applied RoPE transformation.
    """

    def call(self, pos, dim, theta):
        scale = ops.arange(0, dim, 2, dtype="float32") / dim
        omega = 1.0 / (theta**scale)
        out = ops.einsum("...n,d->...nd", pos, omega)
        out = ops.stack(
            [ops.cos(out), -ops.sin(out), ops.sin(out), ops.cos(out)], axis=-1
        )
        out = ops.reshape(out, ops.shape(out)[:-1] + (2, 2))
        return ops.cast(out, dtype="float32")


class ApplyRoPE(keras.layers.Layer):
    """
    Applies the RoPE transformation to the query and key tensors.

    Call arguments:
        xq: KerasTensor. The query tensor of shape (..., L, D).
        xk: KerasTensor. The key tensor of shape (..., L, D).
        freqs_cis: KerasTensor. The frequency complex numbers tensor with shape
            `(..., 2)`.

    Returns:
        tuple[KerasTensor, KerasTensor]: The transformed query and key tensors.
    """

    def call(self, xq, xk, freqs_cis):
        xq_ = ops.reshape(xq, (*ops.shape(xq)[:-1], -1, 1, 2))
        xk_ = ops.reshape(xk, (*ops.shape(xk)[:-1], -1, 1, 2))

        xq_out = (
            freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        )
        xk_out = (
            freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        )

        return ops.reshape(xq_out, ops.shape(xq)), ops.reshape(
            xk_out, ops.shape(xk)
        )


class FluxRoPEAttention(keras.layers.Layer):
    """Computes the attention mechanism with RoPE.

    Args:
        dropout_p: float, optional. Dropout probability. Defaults to 0.0.
        is_causal: bool, optional. If True, applies causal masking. Defaults to
            False.

    Call arguments:
        q: KerasTensor. Query tensor of shape (..., L, D).
        k: KerasTensor. Key tensor of shape (..., S, D).
        v: KerasTensor. Value tensor of shape (..., S, D).
        positional_encoding: KerasTensor. Positional encoding tensor.

    Returns:
        KerasTensor: The resulting tensor from the attention mechanism.
    """

    def __init__(self, dropout_p=0.0, is_causal=False):
        super(FluxRoPEAttention, self).__init__()
        self.dropout_p = dropout_p
        self.is_causal = is_causal

    def call(self, q, k, v, positional_encoding):
        # Apply the RoPE transformation
        q, k = ApplyRoPE()(q, k, positional_encoding)

        # Scaled dot-product attention
        x = scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p, is_causal=self.is_causal
        )
        x = ops.transpose(x, (0, 2, 1, 3))
        b, s, h, d = ops.shape(x)
        return ops.reshape(x, (b, s, h * d))


# TODO: This is probably already implemented in several places, but is needed to
# ensure numeric equivalence to the original implementation. It uses
# torch.functional.scaled_dot_product_attention() - do we have an equivalent
# already in Keras?
def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
):
    """
    Computes the scaled dot-product attention.

    Args:
        query: KerasTensor. Query tensor of shape (..., L, D).
        key: KerasTensor. Key tensor of shape (..., S, D).
        value: KerasTensor. Value tensor of shape (..., S, D).
        attn_mask: KerasTensor, optional. Attention mask tensor. Defaults to
            None.
        dropout_p: float, optional. Dropout probability. Defaults to 0.0.
        is_causal: bool, optional. If True, applies causal masking. Defaults to
            False.
        scale: float, optional. Scale factor for attention. Defaults to None.

    Returns:
        KerasTensor: The output tensor from the attention mechanism.
    """
    L, S = ops.shape(query)[-2], ops.shape(key)[-2]
    scale_factor = (
        1 / ops.sqrt(ops.cast(ops.shape(query)[-1], dtype=query.dtype))
        if scale is None
        else scale
    )
    attn_bias = ops.zeros((L, S), dtype=query.dtype)

    if is_causal:
        assert attn_mask is None
        temp_mask = ops.ones((L, S), dtype=ops.bool)
        temp_mask = ops.tril(temp_mask, diagonal=0)
        attn_bias = ops.where(temp_mask, attn_bias, float("-inf"))

    if attn_mask is not None:
        if ops.shape(attn_mask)[-1] == 1:  # If the mask is 3D
            attn_bias += attn_mask
        else:
            attn_bias = ops.where(attn_mask, attn_bias, float("-inf"))

    # Compute attention weights
    attn_weight = (
        ops.matmul(query, ops.transpose(key, axes=[0, 1, 3, 2])) * scale_factor
    )
    attn_weight += attn_bias
    attn_weight = keras.activations.softmax(attn_weight, axis=-1)

    if dropout_p > 0.0:
        attn_weight = keras.layers.Dropout(dropout_p)(
            attn_weight, training=True
        )

    return ops.matmul(attn_weight, value)


def rearrange_symbolic_tensors(qkv, K, H):
    """
    Splits the qkv tensor into query (q), key (k), and value (v) components.

    Mimics rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=num_heads),
    for graph-mode TensorFlow support when doing functional subclassing
    models.

    Arguments:
        qkv: np.ndarray. Input tensor of shape (B, L, K*H*D).
        K: int. Number of components (q, k, v).
        H: int. Number of attention heads.

    Returns:
        tuple: q, k, v tensors of shape (B, H, L, D).
    """
    # Get the shape of qkv and calculate L and D
    B, L, dim = ops.shape(qkv)
    D = dim // (K * H)

    # Reshape and transpose the qkv tensor
    qkv_reshaped = ops.reshape(qkv, (B, L, K, H, D))
    qkv_transposed = ops.transpose(qkv_reshaped, (2, 0, 3, 1, 4))

    # Split q, k, v along the first dimension (K)
    qkv_splits = ops.split(qkv_transposed, K, axis=0)
    q, k, v = [ops.squeeze(split, 0) for split in qkv_splits]

    return q, k, v
