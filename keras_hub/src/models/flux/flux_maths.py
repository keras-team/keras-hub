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

    def call(self, t, dim=256):
        t = keras.ops.cast(t, "float32")
        half_dim = dim // 2

        exponent = (
            -keras.ops.log(keras.ops.cast(10000.0, "float32"))
            * keras.ops.arange(
                half_dim,
                dtype="float32",
            )
            / keras.ops.cast(
                half_dim,
                "float32",
            )
        )

        emb = keras.ops.exp(exponent)
        emb = t[..., None] * emb[None, ...]
        return keras.ops.concatenate(
            [
                keras.ops.sin(emb),
                keras.ops.cos(emb),
            ],
            axis=-1,
        )


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
        out = ops.stack([ops.cos(out), ops.sin(out)], axis=-1)
        return ops.cast(out, dtype="float32")


class ApplyRoPE(keras.layers.Layer):
    """
    Applies the RoPE transformation to the query and key tensors.

    Call arguments:
        xq: KerasTensor. The query tensor of shape (..., L, D).
        xk: KerasTensor. The key tensor of shape (..., L, D).
        freqs_cis: KerasTensor. The frequency complex numbers tensor with shape
            (..., L, D//2, 2).

    Returns:
        tuple[KerasTensor, KerasTensor]: The transformed query and key tensors.
    """

    def call(self, xq, xk, freqs_cis):
        # xq, xk shape (..., num_heads, seq_len, D)
        # freqs_cis shape (..., seq_len, D//2, 2)
        # Expand freqs_cis to match num_heads dimension
        freqs_cis = ops.expand_dims(freqs_cis, axis=-4)
        # Now freqs_cis shape (..., 1, seq_len, D//2, 2)

        xq_ = ops.reshape(xq, (*ops.shape(xq)[:-1], -1, 2))
        xk_ = ops.reshape(xk, (*ops.shape(xk)[:-1], -1, 2))

        xq_real = xq_[..., 0]
        xq_imag = xq_[..., 1]
        xk_real = xk_[..., 0]
        xk_imag = xk_[..., 1]

        freqs_cos = freqs_cis[..., 0]
        freqs_sin = freqs_cis[..., 1]

        xq_out_real = xq_real * freqs_cos - xq_imag * freqs_sin
        xq_out_imag = xq_real * freqs_sin + xq_imag * freqs_cos
        xk_out_real = xk_real * freqs_cos - xk_imag * freqs_sin
        xk_out_imag = xk_real * freqs_sin + xk_imag * freqs_cos

        xq_out = ops.reshape(
            ops.stack([xq_out_real, xq_out_imag], axis=-1), ops.shape(xq)
        )
        xk_out = ops.reshape(
            ops.stack([xk_out_real, xk_out_imag], axis=-1), ops.shape(xk)
        )

        return xq_out, xk_out


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
    q_shape = query.shape
    k_shape = key.shape

    L = q_shape[-2] if q_shape[-2] is not None else ops.shape(query)[-2]
    S = k_shape[-2] if k_shape[-2] is not None else ops.shape(key)[-2]
    D = q_shape[-1] if q_shape[-1] is not None else ops.shape(query)[-1]

    scale_factor = (
        1 / ops.sqrt(ops.cast(D, dtype=query.dtype)) if scale is None else scale
    )
    attn_bias = ops.zeros((L, S), dtype=query.dtype)

    if is_causal:
        assert attn_mask is None
        temp_mask = ops.ones((L, S), dtype=ops.bool)
        temp_mask = ops.tril(temp_mask, diagonal=0)
        attn_bias = ops.where(temp_mask, attn_bias, float("-inf"))

    if attn_mask is not None:
        if ops.shape(attn_mask)[-1] == 1:
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
    """Splits the qkv tensor into query (q), key (k), and value (v) components.

    Mimics rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=num_heads).

    Arguments:
        qkv: Keras tensor of shape (B, L, K*H*D).
        K: int. Number of components (3 for q, k, v).
        H: int. Number of attention heads.

    Returns:
        tuple: q, k, v tensors of shape (B, H, L, D).
    """
    # Fetch dimension sizes safely (prefer static shape; fall back to ops.shape)
    static_shape = qkv.shape

    b = static_shape[0] if static_shape[0] is not None else ops.shape(qkv)[0]
    l = static_shape[1] if static_shape[1] is not None else ops.shape(qkv)[1]
    dim = static_shape[2] if static_shape[2] is not None else ops.shape(qkv)[2]

    # Compute head dimension D
    d = dim // (K * H)

    # Reshape and transpose: (B, L, K, H, D) -> (K, B, H, L, D)
    qkv_reshaped = ops.reshape(qkv, (b, l, K, H, d))
    qkv_transposed = ops.transpose(qkv_reshaped, (2, 0, 3, 1, 4))

    # Unstack along axis 0 to extract q, k, v
    q = qkv_transposed[0]
    k = qkv_transposed[1]
    v = qkv_transposed[2]

    return q, k, v
