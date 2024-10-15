from dataclasses import dataclass

import keras
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

    Args:
        theta: int. Rotational angle parameter for RoPE.
        axes_dim: list[int]. Dimensionality for each axis of the input tensor.
    """

    def __init__(self, theta: int, axes_dim: list[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.rope = RotaryPositionalEmbedding()

    def build(self, input_shape):
        n_axes = input_shape[-1]
        for i in range(n_axes):
            self.rope.build((input_shape[:-1] + (self.axes_dim[i],)))

    def call(self, ids):
        """
        Computes the positional embeddings for each axis and concatenates them.

        Args:
            ids: KerasTensor. Input tensor of shape (..., num_axes).

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

    Args:
        hidden_dim: int. The dimensionality of the hidden layer.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_layer = layers.Dense(hidden_dim, use_bias=True)
        self.silu = layers.Activation("silu")
        self.out_layer = layers.Dense(hidden_dim, use_bias=True)

    def build(self, input_shape):
        self.in_layer.build(input_shape)
        self.out_layer.build((input_shape[0], self.in_layer.units))

    def call(self, x):
        """
        Applies the MLP embedding to the input tensor.

        Args:
            x: KerasTensor. Input tensor of shape (batch_size, in_dim).

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

    Args:
        dim: int. The dimensionality of the input tensor.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.scale = self.add_weight(
            name="scale", shape=(dim,), initializer="ones"
        )

    def call(self, x):
        """
        Applies RMS normalization to the input tensor.

        Args:
            x: KerasTensor. Input tensor of shape (batch_size, dim).

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

    Args:
        dim: int. The dimensionality of the input query and key tensors.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def build(self, input_shape):
        self.query_norm.build(input_shape)
        self.key_norm.build(input_shape)

    def call(self, q, k):
        """
        Applies RMS normalization to the query and key tensors.

        Args:
            q: KerasTensor. The query tensor of shape (batch_size, dim).
            k: KerasTensor. The key tensor of shape (batch_size, dim).

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

    Args:
        dim: int. Dimensionality of the input tensor.
        num_heads: int. Number of attention heads. Default is 8.
        use_bias: bool. Whether to use bias in the query, key, value projection layers.
            Default is False.
    """

    def __init__(self, dim: int, num_heads: int = 8, use_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        self.qkv = layers.Dense(dim * 3, use_bias=use_bias)
        self.norm = QKNorm(head_dim)
        self.proj = layers.Dense(dim)
        self.attention = FluxRoPEAttention()

    def build(self, input_shape):
        self.qkv.build(input_shape)
        head_dim = input_shape[-1] // self.num_heads
        self.norm.build((None, input_shape[1], head_dim))
        self.proj.build((None, input_shape[1], input_shape[-1]))

    def call(self, x, pe):
        """
        Applies self-attention with RoPE embeddings.

        Args:
            x: KerasTensor. Input tensor of shape (batch_size, seq_len, dim).
            pe: KerasTensor. Positional encoding tensor for RoPE.

        Returns:
            KerasTensor: Output tensor after self-attention and projection.
        """
        qkv = self.qkv(x)

        # Mimics rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        B, L, _ = keras.ops.shape(qkv)
        D = self.hidden_size // self.num_heads

        qkv = keras.ops.reshape(qkv, (B, L, 3, self.num_heads, D))
        qkv = keras.ops.transpose(qkv, (2, 0, 3, 1, 4))
        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

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

    Args:
        dim: int. Dimensionality of the modulation output.
        double: bool. Whether to generate two sets of modulation parameters.
    """

    def __init__(self, dim, double):
        super().__init__()
        self.dim = dim
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = keras.layers.Dense(self.multiplier * dim, use_bias=True)

    def build(self, input_shape):
        self.lin.build(input_shape)

    def call(self, x):
        """
        Generates modulation parameters from the input tensor.

        Args:
            x: KerasTensor. Input tensor.

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
        hidden_size: int. The hidden dimension size for the model.
        num_heads: int. The number of attention heads.
        mlp_ratio: float. The ratio of the MLP hidden dimension to the hidden size.
        use_bias: bool, optional. Whether to include bias in QKV projection. Default is False.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio,
        use_bias=False,
    ):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.image_mod = Modulation(hidden_size, double=True)
        self.image_norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.image_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, use_bias=use_bias
        )

        self.image_norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.image_mlp = keras.Sequential(
            [
                keras.layers.Dense(mlp_hidden_dim, use_bias=True),
                keras.layers.Activation("gelu"),
                keras.layers.Dense(hidden_size, use_bias=True),
            ]
        )

        self.text_mod = Modulation(hidden_size, double=True)
        self.text_norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.text_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, use_bias=use_bias
        )

        self.text_norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.text_mlp = keras.Sequential(
            [
                keras.layers.Dense(mlp_hidden_dim, use_bias=True),
                keras.layers.Activation("gelu"),
                keras.layers.Dense(hidden_size, use_bias=True),
            ]
        )
        self.attention = FluxRoPEAttention()

    def call(self, image, text, vec, pe):
        """
        Forward pass for the DoubleStreamBlock.

        Args:
            image: KerasTensor. Input image tensor.
            text: KerasTensor. Input text tensor.
            vec: KerasTensor. Modulation vector.
            pe: KerasTensor. Positional encoding tensor.

        Returns:
            Tuple[KerasTensor, KerasTensor]: The modified image and text tensors.
        """
        image_mod1, image_mod2 = self.image_mod(vec)
        text_mod1, text_mod2 = self.text_mod(vec)

        # prepare image for attention
        image_modulated = self.image_norm1(image)
        image_modulated = (
            1 + image_mod1.scale
        ) * image_modulated + image_mod1.shift
        image_qkv = self.image_attn.qkv(image_modulated)

        B, L, _ = keras.ops.shape(image_qkv)
        D = self.hidden_size // self.num_heads

        # Mimics rearrange(image_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        image_qkv = keras.ops.reshape(image_qkv, (B, L, 3, self.num_heads, D))
        image_qkv = keras.ops.transpose(image_qkv, (2, 0, 3, 1, 4))
        image_q = image_qkv[:, :, 0]
        image_k = image_qkv[:, :, 1]
        image_v = image_qkv[:, :, 2]
        image_q, image_k = self.image_attn.norm(image_q, image_k)

        # prepare text for attention
        text_modulated = self.text_norm1(text)
        text_modulated = (
            1 + text_mod1.scale
        ) * text_modulated + text_mod1.shift
        text_qkv = self.text_attn.qkv(text_modulated)

        # Mimics rearrange(text_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        text_qkv = keras.ops.reshape(text_qkv, (B, L, 3, self.num_heads, D))
        text_qkv = keras.ops.transpose(text_qkv, (2, 0, 3, 1, 4))

        text_q = text_qkv[:, :, 0]
        text_k = text_qkv[:, :, 1]
        text_v = text_qkv[:, :, 2]

        text_q, text_k = self.text_attn.norm(text_q, text_k)

        # run actual attention
        q = keras.ops.concatenate((text_q, image_q), axis=2)
        k = keras.ops.concatenate((text_k, image_k), axis=2)
        v = keras.ops.concatenate((text_v, image_v), axis=2)

        attn = self.attention(q=q, k=k, v=v, pe=pe)
        text_attn, image_attn = (
            attn[:, : text.shape[1]],
            attn[:, text.shape[1] :],
        )

        # calculate the image blocks
        image = image + image_mod1.gate * self.image_attn.proj(image_attn)
        image = image + image_mod2.gate * self.image_mlp(
            (1 + image_mod2.scale) * self.image_norm2(image) + image_mod2.shift
        )

        # calculate the text blocks
        text = text + text_mod1.gate * self.text_attn.proj(text_attn)
        text = text + text_mod2.gate * self.text_mlp(
            (1 + text_mod2.scale) * self.text_norm2(text) + text_mod2.shift
        )
        return image, text

    def build(self, image_shape, text_shape, vec_shape):
        # Build components for image and text streams
        self.image_mod.build(vec_shape)
        self.image_attn.build(
            (image_shape[0], image_shape[1], self.hidden_size)
        )
        self.image_norm2.build(image_shape)
        self.image_mlp.build(image_shape)

        self.text_mod.build(vec_shape)
        self.text_attn.build((text_shape[0], text_shape[1], self.hidden_size))


class SingleStreamBlock(keras.Model):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.

    Args:
        hidden_size: int. The hidden dimension size for the model.
        num_heads: int. The number of attention heads.
        mlp_ratio: float, optional. The ratio of the MLP hidden dimension to the hidden size. Default is 4.0.
        qk_scale: float, optional. Scaling factor for the query-key product. Default is None.
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

    def build(self, x_shape, vec_shape, pe_shape):
        self.linear1.build(x_shape)
        self.linear2.build(
            (x_shape[0], x_shape[1], self.hidden_size + self.mlp_hidden_dim)
        )

        self.modulation.build(vec_shape)  # Build the modulation layer

        self.norm.build(
            (
                x_shape[0],
                self.num_heads,
                x_shape[1],
                x_shape[-1] // self.num_heads,
            )
        )

    def call(self, x, vec, pe):
        """
        Forward pass for the SingleStreamBlock.

        Args:
            x: KerasTensor. Input tensor.
            vec: KerasTensor. Modulation vector.
            pe: KerasTensor. Positional encoding tensor.

        Returns:
            KerasTensor: The modified input tensor after processing.
        """
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = keras.ops.split(
            self.linear1(x_mod), [3 * self.hidden_size], axis=-1
        )

        # Mimics rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        B, L, _ = keras.ops.shape(qkv)
        D = self.hidden_size // self.num_heads

        qkv = keras.ops.reshape(qkv, (B, L, 3, self.num_heads, D))
        qkv = keras.ops.transpose(qkv, (2, 0, 3, 1, 4))
        
        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

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
        hidden_size: int. The hidden dimension size for the model.
        patch_size: int. The size of each patch.
        output_channels: int. The number of output channels.
    """

    def __init__(self, hidden_size: int, patch_size: int, output_channels: int):
        super().__init__()
        self.norm_final = keras.layers.LayerNormalization(epsilon=1e-6)
        self.linear = keras.layers.Dense(
            patch_size * patch_size * output_channels, use_bias=True
        )
        self.adaLN_modulation = keras.Sequential(
            [
                keras.layers.Activation("silu"),
                keras.layers.Dense(2 * hidden_size, use_bias=True),
            ]
        )

    def build(self, input_shape):
        _, _, features = input_shape

        self.linear.build((None, features))
        self.built = True

    def call(self, x, vec):
        """
        Forward pass for the LastLayer.

        Args:
            x: KerasTensor. Input tensor.
            vec: KerasTensor. Modulation vector.

        Returns:
            KerasTensor: The output tensor after final processing.
        """
        shift, scale = keras.ops.split(self.adaLN_modulation(vec), 2, axis=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
