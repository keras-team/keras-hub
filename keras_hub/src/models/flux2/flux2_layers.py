"""Layers specific to the FLUX.2 rectified flow transformer.

FLUX.2 keeps the overall FLUX.1 MMDiT layout (double-stream blocks followed
by single-stream blocks, multi-axis RoPE, QK RMS-norm, AdaLN modulation), so
the weightless building blocks (`EmbedND`, `QKNorm`, `FluxRoPEAttention`,
`TimestepEmbedding`, ...) are reused from `keras_hub.src.models.flux`. The
layers here capture the architectural differences:

- **Shared modulation.** FLUX.1 gives every block its own modulation
  projection. FLUX.2 computes the AdaLN shift/scale/gate parameters once in
  the backbone (`Flux2Modulation`) and shares them across all double-stream
  blocks (one set for the image stream, one for the text stream) and across
  all single-stream blocks.
- **SwiGLU MLPs.** The feed-forward layers use a SwiGLU gate,
  `silu(x[..., :d]) * x[..., d:]`, instead of a tanh-approximated GELU. The
  gate and up projections are fused into a single `Dense` of width `2 * d`.
- **No biases.** No linear layer in the FLUX.2 transformer has a bias.
"""

import keras
from keras import layers
from keras import ops

from keras_hub.src.models.flux.flux_layers import QKNorm
from keras_hub.src.models.flux.flux_maths import FluxRoPEAttention
from keras_hub.src.models.flux.flux_maths import rearrange_symbolic_tensors


def split_modulation(modulation, num_chunks):
    """Split a fused modulation tensor into per-token broadcastable chunks.

    Args:
        modulation: Tensor of shape `(batch_size, num_chunks * dim)`, as
            produced by `Flux2Modulation`.
        num_chunks: int. The number of chunks, `6` for a double-stream block
            (shift/scale/gate for attention, then for the MLP) or `3` for a
            single-stream block.

    Returns:
        A list of `num_chunks` tensors of shape `(batch_size, 1, dim)`, in
        `shift, scale, gate[, shift, scale, gate]` order.
    """
    return ops.split(modulation[:, None, :], num_chunks, axis=-1)


class Flux2SwiGLU(layers.Layer):
    """SwiGLU gate: `silu(x[..., :d]) * x[..., d:]`.

    FLUX.2 fuses the gate and up projections of every feed-forward block
    into one linear layer, so the gate occupies the *first* half of the
    features and the linear branch the second half. Swapping the halves is
    silent (shapes are identical) but produces a different function.

    Args:
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `trainable`, `dtype` etc.
    """

    def call(self, x):
        gate, value = ops.split(x, 2, axis=-1)
        return keras.activations.silu(gate) * value

    def compute_output_shape(self, input_shape):
        last_dim = input_shape[-1]
        return (
            *input_shape[:-1],
            None if last_dim is None else last_dim // 2,
        )


class Flux2FeedForward(layers.Layer):
    """Bias-free SwiGLU feed-forward block.

    `Dense(2 * mlp_hidden_dim) -> Flux2SwiGLU -> Dense(hidden_size)`.

    Args:
        hidden_size: int. The input and output dimension.
        mlp_hidden_dim: int. The width of the gated hidden layer.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `trainable`, `dtype` etc.
    """

    def __init__(self, hidden_size, mlp_hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.mlp_hidden_dim = mlp_hidden_dim

        self.input_layer = layers.Dense(
            2 * mlp_hidden_dim, use_bias=False, name="input_layer"
        )
        self.swiglu = Flux2SwiGLU(name="swiglu")
        self.output_layer = layers.Dense(
            hidden_size, use_bias=False, name="output_layer"
        )

    def build(self, input_shape):
        self.input_layer.build(input_shape)
        self.output_layer.build((*input_shape[:-1], self.mlp_hidden_dim))

    def call(self, x):
        x = self.input_layer(x)
        x = self.swiglu(x)
        return self.output_layer(x)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.hidden_size)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "mlp_hidden_dim": self.mlp_hidden_dim,
            }
        )
        return config


class Flux2MLPEmbedder(layers.Layer):
    """Bias-free MLP that embeds the sinusoidal timestep (and guidance).

    `Dense(hidden_dim) -> SiLU -> Dense(hidden_dim)`, matching
    `time_in` / `guidance_in` in the reference checkpoint.

    Args:
        hidden_dim: int. The dimensionality of both projections.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `trainable`, `dtype` etc.
    """

    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.input_layer = layers.Dense(
            hidden_dim, use_bias=False, name="input_layer"
        )
        self.output_layer = layers.Dense(
            hidden_dim, use_bias=False, name="output_layer"
        )

    def build(self, input_shape):
        self.input_layer.build(input_shape)
        self.output_layer.build((*input_shape[:-1], self.hidden_dim))

    def call(self, x):
        x = self.input_layer(x)
        x = keras.activations.silu(x)
        return self.output_layer(x)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.hidden_dim)

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config


class Flux2Modulation(layers.Layer):
    """Shared AdaLN modulation projection.

    Computes `Dense(multiplier * dim)(silu(x))` without a bias, where
    `multiplier` is `6` for double-stream blocks and `3` for single-stream
    blocks. The fused output is returned as is and split inside each block
    with `split_modulation`, so a single `Flux2Modulation` can feed every
    block of one kind.

    Args:
        dim: int. The hidden size of the transformer.
        double: bool. Whether to produce the two parameter sets used by a
            double-stream block (attention and MLP) or the single set used
            by a single-stream block.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `trainable`, `dtype` etc.
    """

    def __init__(self, dim, double, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.double = double
        self.multiplier = 6 if double else 3
        self.linear_projection = layers.Dense(
            self.multiplier * dim, use_bias=False, name="linear_projection"
        )

    def build(self, input_shape):
        self.linear_projection.build(input_shape)

    def call(self, x):
        return self.linear_projection(keras.activations.silu(x))

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.multiplier * self.dim)

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "double": self.double})
        return config


class Flux2DoubleStreamBlock(layers.Layer):
    """FLUX.2 double-stream block.

    The image and text streams have separate weights but attend jointly
    over the concatenated `[text, image]` sequence. Unlike FLUX.1, the AdaLN
    parameters are not computed here: they are passed in, already projected
    by the backbone's shared `Flux2Modulation` layers.

    Args:
        hidden_size: int. The hidden size of the transformer.
        num_heads: int. The number of attention heads.
        mlp_ratio: float. The ratio of the MLP hidden size to `hidden_size`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `trainable`, `dtype` etc.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        head_dim = hidden_size // num_heads

        # Image stream.
        self.image_norm1 = layers.LayerNormalization(
            epsilon=1e-6, scale=False, center=False, name="image_norm1"
        )
        self.image_qkv = layers.Dense(
            3 * hidden_size, use_bias=False, name="image_qkv"
        )
        self.image_attn_norm = QKNorm(head_dim, name="image_attn_norm")
        self.image_attn_proj = layers.Dense(
            hidden_size, use_bias=False, name="image_attn_proj"
        )
        self.image_norm2 = layers.LayerNormalization(
            epsilon=1e-6, scale=False, center=False, name="image_norm2"
        )
        self.image_mlp = Flux2FeedForward(
            hidden_size, self.mlp_hidden_dim, name="image_mlp"
        )

        # Text stream.
        self.text_norm1 = layers.LayerNormalization(
            epsilon=1e-6, scale=False, center=False, name="text_norm1"
        )
        self.text_qkv = layers.Dense(
            3 * hidden_size, use_bias=False, name="text_qkv"
        )
        self.text_attn_norm = QKNorm(head_dim, name="text_attn_norm")
        self.text_attn_proj = layers.Dense(
            hidden_size, use_bias=False, name="text_attn_proj"
        )
        self.text_norm2 = layers.LayerNormalization(
            epsilon=1e-6, scale=False, center=False, name="text_norm2"
        )
        self.text_mlp = Flux2FeedForward(
            hidden_size, self.mlp_hidden_dim, name="text_mlp"
        )

        self.attention = FluxRoPEAttention(name="attention")

    def build(
        self,
        image_shape,
        text_shape,
        image_modulation_shape,
        text_modulation_shape,
        positional_encoding_shape,
    ):
        # Built explicitly because `compute_output_spec` below stops Keras
        # from tracing `call` while constructing the functional graph, so
        # nothing else would create the sublayer weights.
        head_dim = self.hidden_size // self.num_heads
        for prefix, shape in (("image", image_shape), ("text", text_shape)):
            qk_shape = (shape[0], self.num_heads, shape[1], head_dim)
            getattr(self, f"{prefix}_norm1").build(shape)
            getattr(self, f"{prefix}_qkv").build(shape)
            getattr(self, f"{prefix}_attn_norm").build(qk_shape)
            getattr(self, f"{prefix}_attn_proj").build(shape)
            getattr(self, f"{prefix}_norm2").build(shape)
            getattr(self, f"{prefix}_mlp").build(shape)

    def compute_output_spec(
        self,
        image,
        text,
        image_modulation,
        text_modulation,
        positional_encoding,
    ):
        # The streams are residual, so their shapes are unchanged. Declaring
        # this avoids tracing `call` with independent placeholder sequence
        # lengths, which cannot satisfy the RoPE broadcast when the sequence
        # axes are dynamic.
        return (
            keras.KerasTensor(image.shape, dtype=self.compute_dtype),
            keras.KerasTensor(text.shape, dtype=self.compute_dtype),
        )

    def call(
        self,
        image,
        text,
        image_modulation,
        text_modulation,
        positional_encoding,
    ):
        (
            img_shift1,
            img_scale1,
            img_gate1,
            img_shift2,
            img_scale2,
            img_gate2,
        ) = split_modulation(image_modulation, 6)
        (
            txt_shift1,
            txt_scale1,
            txt_gate1,
            txt_shift2,
            txt_scale2,
            txt_gate2,
        ) = split_modulation(text_modulation, 6)

        img_normed = self.image_norm1(image) * (1 + img_scale1) + img_shift1
        txt_normed = self.text_norm1(text) * (1 + txt_scale1) + txt_shift1

        img_q, img_k, img_v = rearrange_symbolic_tensors(
            self.image_qkv(img_normed), 3, self.num_heads
        )
        txt_q, txt_k, txt_v = rearrange_symbolic_tensors(
            self.text_qkv(txt_normed), 3, self.num_heads
        )
        img_q, img_k = self.image_attn_norm(img_q, img_k)
        txt_q, txt_k = self.text_attn_norm(txt_q, txt_k)

        # Text first, to match the `[text_ids, image_ids]` RoPE positions
        # built by `Flux2Backbone`.
        q = ops.concatenate([txt_q, img_q], axis=2)
        k = ops.concatenate([txt_k, img_k], axis=2)
        v = ops.concatenate([txt_v, img_v], axis=2)
        attn = self.attention(q, k, v, positional_encoding)

        txt_len = text.shape[1]
        if txt_len is None:
            txt_len = ops.shape(text)[1]
        txt_attn = attn[:, :txt_len, :]
        img_attn = attn[:, txt_len:, :]

        image = image + img_gate1 * self.image_attn_proj(img_attn)
        text = text + txt_gate1 * self.text_attn_proj(txt_attn)

        img_normed = self.image_norm2(image) * (1 + img_scale2) + img_shift2
        txt_normed = self.text_norm2(text) * (1 + txt_scale2) + txt_shift2
        image = image + img_gate2 * self.image_mlp(img_normed)
        text = text + txt_gate2 * self.text_mlp(txt_normed)
        return image, text

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
            }
        )
        return config


class Flux2SingleStreamBlock(layers.Layer):
    """FLUX.2 single-stream ("parallel") block.

    The attention QKV projection is fused with the SwiGLU gate/up projection
    in `linear1`, and the attention output projection is fused with the MLP
    down projection in `linear2`, as in ViT-22B
    (https://arxiv.org/abs/2302.05442). The AdaLN parameters are passed in
    from the backbone's shared `Flux2Modulation` layer.

    Args:
        hidden_size: int. The hidden size of the transformer.
        num_heads: int. The number of attention heads.
        mlp_ratio: float. The ratio of the MLP hidden size to `hidden_size`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `trainable`, `dtype` etc.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        head_dim = hidden_size // num_heads

        self.pre_norm = layers.LayerNormalization(
            epsilon=1e-6, scale=False, center=False, name="pre_norm"
        )
        # [q, k, v, mlp_gate, mlp_up].
        self.linear1 = layers.Dense(
            3 * hidden_size + 2 * self.mlp_hidden_dim,
            use_bias=False,
            name="linear1",
        )
        # [attention, mlp] -> hidden.
        self.linear2 = layers.Dense(hidden_size, use_bias=False, name="linear2")
        self.norm = QKNorm(head_dim, name="norm")
        self.mlp_activation = Flux2SwiGLU(name="mlp_activation")
        self.attention = FluxRoPEAttention(name="attention")

    def build(self, x_shape, modulation_shape, positional_encoding_shape):
        head_dim = self.hidden_size // self.num_heads
        self.pre_norm.build(x_shape)
        self.linear1.build(x_shape)
        self.linear2.build(
            (*x_shape[:-1], self.hidden_size + self.mlp_hidden_dim)
        )
        self.norm.build((x_shape[0], self.num_heads, x_shape[1], head_dim))

    def compute_output_spec(self, x, modulation, positional_encoding):
        # Residual block: the shape is unchanged. See
        # `Flux2DoubleStreamBlock.compute_output_spec`.
        return keras.KerasTensor(x.shape, dtype=self.compute_dtype)

    def call(self, x, modulation, positional_encoding):
        shift, scale, gate = split_modulation(modulation, 3)
        x_mod = (1 + scale) * self.pre_norm(x) + shift

        qkv, mlp = ops.split(
            self.linear1(x_mod), [3 * self.hidden_size], axis=-1
        )
        q, k, v = rearrange_symbolic_tensors(qkv, K=3, H=self.num_heads)
        q, k = self.norm(q, k)
        attn = self.attention(q, k, v, positional_encoding)

        output = self.linear2(
            ops.concatenate([attn, self.mlp_activation(mlp)], axis=-1)
        )
        return x + gate * output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
            }
        )
        return config


class Flux2LastLayer(layers.Layer):
    """Final AdaLN + projection, without biases.

    `shift, scale = Dense(2 * hidden_size)(silu(vec))`, applied to a
    non-affine LayerNorm, followed by a projection to `output_channels`.
    The `(shift, scale)` order matches the reference checkpoint's
    `final_layer.adaLN_modulation`.

    Args:
        hidden_size: int. The hidden size of the transformer.
        output_channels: int. The number of output channels per token.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `trainable`, `dtype` etc.
    """

    def __init__(self, hidden_size, output_channels, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.output_channels = output_channels

        self.norm_final = layers.LayerNormalization(
            epsilon=1e-6, scale=False, center=False, name="norm_final"
        )
        self.adaLN_modulation = layers.Dense(
            2 * hidden_size, use_bias=False, name="adaLN_modulation"
        )
        self.linear = layers.Dense(
            output_channels, use_bias=False, name="linear"
        )

    def build(self, x_shape, modulation_encoding_shape):
        self.norm_final.build(x_shape)
        self.adaLN_modulation.build(modulation_encoding_shape)
        self.linear.build(x_shape)

    def call(self, x, modulation_encoding):
        shift, scale = ops.split(
            self.adaLN_modulation(keras.activations.silu(modulation_encoding)),
            2,
            axis=-1,
        )
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        return self.linear(x)

    def compute_output_shape(self, x_shape, modulation_encoding_shape):
        return (*x_shape[:-1], self.output_channels)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "output_channels": self.output_channels,
            }
        )
        return config
