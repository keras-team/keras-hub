# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import keras
from einops import rearrange
from keras import ops


class TimestepEmbedding(keras.layers.Layer):
    """
    Creates sinusoidal timestep embeddings.


    Call Args:
        t (KerasTensor): A 1-D tensor of shape (N,), representing N indices, one per batch element.
                         These values may be fractional.
        dim (int): The dimension of the output.
        max_period (int, optional): Controls the minimum frequency of the embeddings. Defaults to 10000.
        time_factor (float, optional): A scaling factor applied to `t`. Defaults to 1000.0.

    Returns:
        KerasTensor: A tensor of shape (N, D) representing the positional embeddings,
                     where N is the number of batch elements and D is the specified dimension `dim`.
    """

    def call(self, t, dim, max_period=10000, time_factor=1000.0):
        t = time_factor * t
        half_dim = dim // 2
        freqs = ops.exp(
            -ops.log(max_period)
            * ops.arange(half_dim, dtype="float32")
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


    Call Args:
        pos (KerasTensor): The positional tensor with shape (..., n, d).
        dim (int): The embedding dimension, should be even.
        theta (int): The base frequency.

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
        out = rearrange(out, "... n d (i j) -> ... n d i j", i=2, j=2)
        return ops.cast(out, dtype="float32")


class ApplyRoPE(keras.layers.Layer):
    """
    Applies the RoPE transformation to the query and key tensors.

    Call Args:
        xq (KerasTensor): The query tensor of shape (..., L, D).
        xk (KerasTensor): The key tensor of shape (..., L, D).
        freqs_cis (KerasTensor): The frequency complex numbers tensor with shape (..., 2).

    Returns:
        tuple[KerasTensor, KerasTensor]: The transformed query and key tensors.
    """

    def call(self, xq, xk, freqs_cis):
        xq_ = ops.reshape(
            ops.cast(xq, "float32"), (*ops.shape(xq)[:-1], -1, 1, 2)
        )
        xk_ = ops.reshape(
            ops.cast(xk, "float32"), (*ops.shape(xk)[:-1], -1, 1, 2)
        )

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
    """
    Computes the attention mechanism with the RoPE transformation applied to the query and key tensors.

    Args:
        dropout_p (float, optional): Dropout probability. Defaults to 0.0.
        is_causal (bool, optional): If True, applies causal masking. Defaults to False.

    Call Args:
        q (KerasTensor): Query tensor of shape (..., L, D).
        k (KerasTensor): Key tensor of shape (..., S, D).
        v (KerasTensor): Value tensor of shape (..., S, D).
        pe (KerasTensor): Positional encoding tensor.

    Returns:
        KerasTensor: The resulting tensor from the attention mechanism.
    """

    def __init__(self, dropout_p=0.0, is_causal=False):
        super(FluxRoPEAttention, self).__init__()
        self.dropout_p = dropout_p
        self.is_causal = is_causal

    def call(self, q, k, v, pe):
        # Apply the RoPE transformation
        q, k = ApplyRoPE()(q, k, pe)

        # Scaled dot-product attention
        x = scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p, is_causal=self.is_causal
        )

        # Reshape the output
        B, H, L, D = (
            ops.shape(x)[0],
            ops.shape(x)[1],
            ops.shape(x)[2],
            ops.shape(x)[3],
        )
        x = rearrange(x, "B H L D -> B L (H D)")
        return x


# TODO: This is probably already implemented in several places, but is needed to ensure numeric equivalence to the original
# implementation. It uses torch.functional.scaled_dot_product_attention() - do we have an equivalent already in Keras?
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
        query (KerasTensor): Query tensor of shape (..., L, D).
        key (KerasTensor): Key tensor of shape (..., S, D).
        value (KerasTensor): Value tensor of shape (..., S, D).
        attn_mask (KerasTensor, optional): Attention mask tensor. Defaults to None.
        dropout_p (float, optional): Dropout probability. Defaults to 0.0.
        is_causal (bool, optional): If True, applies causal masking. Defaults to False.
        scale (float, optional): Scale factor for attention. Defaults to None.

    Returns:
        KerasTensor: The output tensor from the attention mechanism.
    """
    L, S = ops.shape(query)[-2], ops.shape(key)[-2]
    scale_factor = (
        1 / ops.sqrt(ops.cast(ops.shape(query)[-1], "float32"))
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
