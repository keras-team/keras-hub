# Copyright 2024 The KerasHub Authors
#
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

from dataclasses import dataclass

import keras
from einops import rearrange
from keras import KerasTensor
from keras import layers
from keras import ops

from keras_hub.src.models.flux.flux_maths import FluxRoPEAttention
from keras_hub.src.models.flux.flux_maths import RotaryPositionalEmbedding


class EmbedND(keras.Model):
    """
    Embedding layer for N-dimensional inputs using Rotary Positional Embedding (RoPE).

    This layer applies RoPE embeddings across multiple axes of the input tensor and
    concatenates the embeddings along a specified axis.
    """

    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        """
        Initializes the EmbedND layer.

        Args:
            dim (int): Dimensionality of the embedding.
            theta (int): Rotational angle parameter for RoPE.
            axes_dim (list[int]): Dimensionality for each axis of the input tensor.
        """
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim
        self.rope = RotaryPositionalEmbedding()

    def call(self, ids):
        """
        Computes the positional embeddings for each axis and concatenates them.

        Args:
            ids (KerasTensor): Input tensor of shape (..., num_axes).

        Returns:
            KerasTensor: Positional embeddings of shape (..., concatenated_dim, 1, ...).
        """
        n_axes = ids.shape[-1]
        emb = keras.ops.concatenate(
            [
                self.rope(ids[..., i], dim=self.axes_dim[i], theta=self.theta)
                for i in range(n_axes)
            ],
            axis=-3,
        )

        return keras.ops.expand_dims(emb, axis=1)


class MLPEmbedder(keras.Model):
    """
    A simple multi-layer perceptron (MLP) embedder model.

    This model applies a linear transformation followed by the SiLU activation
    function and another linear transformation to the input tensor.
    """

    def __init__(self, hidden_dim: int):
        """
        Initializes the MLPEmbedder.

        Args:
            hidden_dim (int): The dimensionality of the hidden layer.
        """
        super().__init__()
        self.in_layer = layers.Dense(hidden_dim, use_bias=True)
        self.silu = layers.Activation("silu")
        self.out_layer = layers.Dense(hidden_dim, use_bias=True)

    def call(self, x: KerasTensor) -> KerasTensor:
        """
        Applies the MLP embedding to the input tensor.

        Args:
            x (KerasTensor): Input tensor of shape (batch_size, in_dim).

        Returns:
            KerasTensor: Output tensor of shape (batch_size, hidden_dim) after applying
            the MLP transformations.
        """
        x = self.in_layer(x)
        x = self.silu(x)
        return self.out_layer(x)


# TODO: Maybe this can be exported as part of the public API? Seems to have enough reusability.
class RMSNorm(keras.layers.Layer):
    """
    Root Mean Square (RMS) Normalization layer.

    This layer normalizes the input tensor based on its RMS value and applies
    a learned scaling factor.
    """

    def __init__(self, dim: int):
        """
        Initializes the RMSNorm layer.

        Args:
            dim (int): The dimensionality of the input tensor.
        """
        super().__init__()
        self.scale = self.add_weight(
            name="scale", shape=(dim,), initializer="ones"
        )

    def call(self, x: KerasTensor) -> KerasTensor:
        """
        Applies RMS normalization to the input tensor.

        Args:
            x (KerasTensor): Input tensor of shape (batch_size, dim).

        Returns:
            KerasTensor: The RMS-normalized tensor of the same shape (batch_size, dim),
            scaled by the learned `scale` parameter.
        """
        x = ops.cast(x, float)
        rrms = ops.rsqrt(ops.mean(ops.square(x), axis=-1, keepdims=True) + 1e-6)
        return (x * rrms) * self.scale


class QKNorm(keras.layers.Layer):
    """
    A layer that applies RMS normalization to query and key tensors.

    This layer normalizes the input query and key tensors using separate RMSNorm
    layers for each.
    """

    def __init__(self, dim: int):
        """
        Initializes the QKNorm layer.

        Args:
            dim (int): The dimensionality of the input query and key tensors.
        """
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def call(
        self, q: KerasTensor, k: KerasTensor
    ) -> tuple[KerasTensor, KerasTensor]:
        """
        Applies RMS normalization to the query and key tensors.

        Args:
            q (KerasTensor): The query tensor of shape (batch_size, dim).
            k (KerasTensor): The key tensor of shape (batch_size, dim).

        Returns:
            tuple[KerasTensor, KerasTensor]: A tuple containing the normalized query and key tensors.
        """
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q, k


class SelfAttention(keras.Model):
    """
    Multi-head self-attention layer with RoPE embeddings and RMS normalization.

    This layer performs self-attention over the input sequence and applies RMS
    normalization to the query and key tensors before computing the attention scores.
    """

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        """
        Initializes the SelfAttention layer.

        Args:
            dim (int): Dimensionality of the input tensor.
            num_heads (int): Number of attention heads. Default is 8.
            qkv_bias (bool): Whether to use bias in the query, key, value projection layers.
                Default is False.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = layers.Dense(dim)
        self.attention = FluxRoPEAttention()

    def call(self, x, pe):
        """
        Applies self-attention with RoPE embeddings.

        Args:
            x (KerasTensor): Input tensor of shape (batch_size, seq_len, dim).
            pe (KerasTensor): Positional encoding tensor for RoPE.

        Returns:
            KerasTensor: Output tensor after self-attention and projection.
        """
        qkv = self.qkv(x)
        q, k, v = rearrange(
            qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        q, k = self.norm(q, k)
        x = self.attention(q=q, k=k, v=v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: KerasTensor
    scale: KerasTensor
    gate: KerasTensor


class Modulation(keras.Model):
    """
    Modulation layer that produces shift, scale, and gate tensors.

    This layer applies a SiLU activation to the input tensor followed by a linear
    transformation to generate modulation parameters. It can optionally generate two
    sets of modulation parameters.
    """

    def __init__(self, dim, double):
        """
        Initializes the Modulation layer.

        Args:
            dim (int): Dimensionality of the modulation output.
            double (bool): Whether to generate two sets of modulation parameters.
        """
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = keras.layers.Dense(self.multiplier * dim, use_bias=True)

    def call(self, x):
        """
        Generates modulation parameters from the input tensor.

        Args:
            x (KerasTensor): Input tensor.

        Returns:
            tuple[ModulationOut, ModulationOut | None]: A tuple containing the shift,
            scale, and gate tensors. If `double` is True, returns two sets of modulation parameters.
        """
        x = keras.layers.Activation("silu")(x)
        out = self.lin(x)
        out = keras.ops.split(
            out[:, None, :], indices_or_sections=self.multiplier, axis=-1
        )

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(keras.Model):
    """
    A block that processes image and text inputs in parallel using
    self-attention and MLP layers, with modulation.

    Args:
        hidden_size (int): The hidden dimension size for the model.
        num_heads (int): The number of attention heads.
        mlp_ratio (float): The ratio of the MLP hidden dimension to the hidden size.
        qkv_bias (bool, optional): Whether to include bias in QKV projection. Default is False.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool = False,
    ):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.img_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias
        )

        self.img_norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.img_mlp = keras.Sequential(
            [
                keras.layers.Dense(mlp_hidden_dim, use_bias=True),
                keras.layers.Activation("gelu"),
                keras.layers.Dense(hidden_size, use_bias=True),
            ]
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.txt_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias
        )

        self.txt_norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.txt_mlp = keras.Sequential(
            [
                keras.layers.Dense(mlp_hidden_dim, use_bias=True),
                keras.layers.Activation("gelu"),
                keras.layers.Dense(hidden_size, use_bias=True),
            ]
        )
        self.attention = FluxRoPEAttention()

    def call(self, img, txt, vec, pe):
        """
        Forward pass for the DoubleStreamBlock.

        Args:
            img (KerasTensor): Input image tensor.
            txt (KerasTensor): Input text tensor.
            vec (KerasTensor): Modulation vector.
            pe (KerasTensor): Positional encoding tensor.

        Returns:
            Tuple[KerasTensor, KerasTensor]: The modified image and text tensors.
        """
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        img_q, img_k = self.img_attn.norm(img_q, img_k)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k)

        # run actual attention
        q = keras.ops.concatenate((txt_q, img_q), axis=2)
        k = keras.ops.concatenate((txt_k, img_k), axis=2)
        v = keras.ops.concatenate((txt_v, img_v), axis=2)

        attn = self.attention(q=q, k=k, v=v, pe=pe)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp(
            (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
        )

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp(
            (1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift
        )
        return img, txt


class SingleStreamBlock(keras.Model):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.

    Args:
        hidden_size (int): The hidden dimension size for the model.
        num_heads (int): The number of attention heads.
        mlp_ratio (float, optional): The ratio of the MLP hidden dimension to the hidden size. Default is 4.0.
        qk_scale (float, optional): Scaling factor for the query-key product. Default is None.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = keras.layers.Dense(hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = keras.layers.Dense(hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.modulation = Modulation(hidden_size, double=False)
        self.attention = FluxRoPEAttention()

    def call(self, x, vec, pe):
        """
        Forward pass for the SingleStreamBlock.

        Args:
            x (KerasTensor): Input tensor.
            vec (KerasTensor): Modulation vector.
            pe (KerasTensor): Positional encoding tensor.

        Returns:
            KerasTensor: The modified input tensor after processing.
        """
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = keras.ops.split(
            self.linear1(x_mod), [3 * self.hidden_size], axis=-1
        )

        q, k, v = rearrange(
            qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        q, k = self.norm(q, k)

        # compute attention
        attn = self.attention(q, k=k, v=v, pe=pe)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(
            keras.ops.concatenate(
                (attn, keras.activations.gelu(mlp, approximate=True)), 2
            )
        )
        return x + mod.gate * output


class LastLayer(keras.Model):
    """
    Final layer for processing output tensors with adaptive normalization.

    Args:
        hidden_size (int): The hidden dimension size for the model.
        patch_size (int): The size of each patch.
        out_channels (int): The number of output channels.
    """

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = keras.layers.LayerNormalization(epsilon=1e-6)
        self.linear = keras.layers.Dense(
            patch_size * patch_size * out_channels, use_bias=True
        )
        self.adaLN_modulation = keras.Sequential(
            [
                keras.layers.Activation("silu"),
                keras.layers.Dense(2 * hidden_size, use_bias=True),
            ]
        )

    def call(self, x, vec):
        """
        Forward pass for the LastLayer.

        Args:
            x (KerasTensor): Input tensor.
            vec (KerasTensor): Modulation vector.

        Returns:
            KerasTensor: The output tensor after final processing.
        """
        shift, scale = keras.ops.split(self.adaLN_modulation(vec), 2, axis=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
