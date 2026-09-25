import keras
from keras import layers
from keras import ops

from keras_hub.src.layers.modeling.rms_normalization import RMSNormalization
from keras_hub.src.models.flux.flux_maths import FluxRoPEAttention
from keras_hub.src.models.flux.flux_maths import RotaryPositionalEmbedding
from keras_hub.src.models.flux.flux_maths import rearrange_symbolic_tensors


class ApproximateGELU(layers.Layer):
    """GELU using the tanh approximation.

    FLUX uses `nn.GELU(approximate="tanh")` in the double-stream MLPs
    (`activation_fn="gelu-approximate"` in diffusers). Keras's
    `Activation("gelu")` is the *exact*, erf-based formulation, which is a
    different function -- close enough to look plausible, but it perturbs
    every double-block MLP output and compounds across the 19 blocks.
    """

    def call(self, inputs):
        return keras.activations.gelu(inputs, approximate=True)


class StripTextTokens(layers.Layer):
    """Drop the leading text tokens from a fused `[text, image]` sequence.

    The single-stream blocks run on `concat([text, image])`, so the image
    tokens have to be sliced back out afterwards. That slice cannot be done
    while building the functional graph when the sequence axes are dynamic:
    `KerasTensor.shape[1]` is `None` there (and `ops.shape` returns the same
    static `None`), so `sequence[:, None:, ...]` is a silent no-op that
    leaves the text tokens in the output. Performing it inside a layer
    defers it to a context where the true runtime length is known.
    """

    def call(self, sequence, text):
        return sequence[:, ops.shape(text)[1] :, ...]

    def compute_output_spec(self, sequence, text):
        shape = list(sequence.shape)
        if shape[1] is not None and text.shape[1] is not None:
            shape[1] = shape[1] - text.shape[1]
        else:
            shape[1] = None
        return keras.KerasTensor(tuple(shape), dtype=self.compute_dtype)


class EmbedND(keras.Model):
    """Embedding layer for N-dimensional inputs using RoPE.

    This layer applies RoPE embeddings across multiple axes of the input tensor
    and concatenates the embeddings along a specified axis.

    Args:
        theta. Rotational angle parameter for RoPE.
        axes_dim. Dimensionality for each axis of the input tensor.
    """

    def __init__(self, theta, axes_dim):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.rope = RotaryPositionalEmbedding()

    def build(self, input_shape):
        n_axes = input_shape[-1]
        if n_axes != len(self.axes_dim):
            raise ValueError(
                f"EmbedND received {n_axes} positional axes, "
                f"but axes_dim has {len(self.axes_dim)} entries. "
                f"input_shape={input_shape}, axes_dim={self.axes_dim}"
            )

        for i in range(n_axes):
            self.rope.build(input_shape[:-1] + (self.axes_dim[i],))

    def call(self, ids):
        """Computes the positional embeddings for each axis and concatenates.

        Args:
            ids: KerasTensor. Input tensor of shape (..., num_axes).

        Returns:
            KerasTensor: Positional embeddings of shape
            (..., sum(axes_dim) // 2, 2).
        """
        n_axes = ids.shape[-1]
        emb = ops.concatenate(
            [
                self.rope(ids[..., i], dim=self.axes_dim[i], theta=self.theta)
                for i in range(n_axes)
            ],
            axis=-2,
        )

        return emb


class MLPEmbedder(keras.Model):
    """A simple multi-layer perceptron (MLP) embedder model.

    This model applies a linear transformation followed by the SiLU activation
    function and another linear transformation to the input tensor.

    Args:
        hidden_dim. The dimensionality of the hidden layer.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_layer = layers.Dense(hidden_dim, use_bias=True)
        self.silu = layers.Activation("silu")
        self.output_layer = layers.Dense(hidden_dim, use_bias=True)

    def build(self, input_shape):
        self.input_layer.build(input_shape)
        self.output_layer.build((input_shape[0], self.input_layer.units))

    def call(self, x):
        """Applies the MLP embedding to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, in_dim).

        Returns:
            Output tensor of shape (batch_size, hidden_dim) after applying the
            MLP transformations.
        """
        x = self.input_layer(x)
        x = self.silu(x)
        return self.output_layer(x)


class QKNorm(keras.layers.Layer):
    """A layer that applies RMS normalization to query and key tensors.

    This layer normalizes the input query and key tensors using separate
    RMSNormalization layers for each.

    Args:
        input_dim. The dimensionality of the input query and key tensors.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.query_norm = RMSNormalization(input_dim)
        self.key_norm = RMSNormalization(input_dim)

    def build(self, input_shape):
        self.query_norm.build(input_shape)
        self.key_norm.build(input_shape)

    def call(self, q, k):
        """
        Applies RMS normalization to the query and key tensors.

        Args:
            q: KerasTensor. The query tensor of shape (batch_size, input_dim).
            k: KerasTensor. The key tensor of shape (batch_size, input_dim).

        Returns:
            tuple[KerasTensor, KerasTensor]: A tuple containing the normalized
            query and key tensors.
        """
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q, k


class Modulation(keras.Model):
    """Modulation layer that produces shift, scale, and gate tensors.

    This layer applies a SiLU activation to the input tensor followed by a
    linear transformation to generate modulation parameters. It can optionally
    generate two sets of modulation parameters.

    Args:
        dim: int. Dimensionality of the modulation output.
        double: bool. Whether to generate two sets of modulation parameters.
    """

    def __init__(self, dim, double):
        super().__init__()
        self.dim = dim
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.linear_projection = keras.layers.Dense(
            self.multiplier * dim, use_bias=True
        )

    def build(self, input_shape):
        self.linear_projection.build(input_shape)

    def call(self, x):
        """
        Generates modulation parameters from the input tensor.

        Args:
            x: KerasTensor. Input tensor.

        Returns:
            tuple[ModulationOut, ModulationOut | None]: A tuple containing th
            shift, scale, and gate tensors. If `double` is True, returns two
            sets of modulation parameters.
        """
        x = keras.layers.Activation("silu")(x)
        out = self.linear_projection(x)
        out = ops.split(
            out[:, None, :], indices_or_sections=self.multiplier, axis=-1
        )

        first_output = {"shift": out[0], "scale": out[1], "gate": out[2]}
        second_output = (
            {"shift": out[3], "scale": out[4], "gate": out[5]}
            if self.is_double
            else None
        )

        return first_output, second_output


class DoubleStreamBlock(keras.layers.Layer):
    """
    A block that processes image and text inputs in parallel using
    self-attention and MLP layers, with modulation.

    Args:
        hidden_size: int. The hidden dimension size for the model.
        num_heads: int. The number of attention heads.
        mlp_ratio: float. The ratio of the MLP hidden dimension to the hidde
            size.
        use_bias: bool, optional. Whether to include bias in QKV projection.
            Default is False.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio,
        use_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_bias = use_bias
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        head_dim = hidden_size // num_heads

        # Image stream layers
        self.image_mod = Modulation(hidden_size, double=True)
        self.image_norm1 = keras.layers.LayerNormalization(
            epsilon=1e-6, scale=False, center=False
        )
        self.image_qkv = keras.layers.Dense(
            3 * hidden_size, use_bias=use_bias, name="img_qkv"
        )
        # Maps to `double_blocks.{i}.img_attn.norm.*` in the reference
        # checkpoint. Without this the q/k RMS scales are silently dropped.
        self.image_attn_norm = QKNorm(head_dim)
        self.image_attn_proj = keras.layers.Dense(
            hidden_size, use_bias=use_bias, name="img_attn_proj"
        )

        self.image_norm2 = keras.layers.LayerNormalization(
            epsilon=1e-6, scale=False, center=False
        )
        self.image_mlp = keras.Sequential(
            [
                keras.layers.Dense(mlp_hidden_dim, use_bias=True),
                ApproximateGELU(),
                keras.layers.Dense(hidden_size, use_bias=True),
            ],
            name="image_mlp",
        )

        # Text stream layers
        self.text_mod = Modulation(hidden_size, double=True)
        self.text_norm1 = keras.layers.LayerNormalization(
            epsilon=1e-6, scale=False, center=False
        )
        self.text_qkv = keras.layers.Dense(
            3 * hidden_size, use_bias=use_bias, name="txt_qkv"
        )
        # Maps to `double_blocks.{i}.txt_attn.norm.*`.
        self.text_attn_norm = QKNorm(head_dim)
        self.text_attn_proj = keras.layers.Dense(
            hidden_size, use_bias=use_bias, name="txt_attn_proj"
        )

        self.text_norm2 = keras.layers.LayerNormalization(
            epsilon=1e-6, scale=False, center=False
        )
        self.text_mlp = keras.Sequential(
            [
                keras.layers.Dense(mlp_hidden_dim, use_bias=True),
                ApproximateGELU(),
                keras.layers.Dense(hidden_size, use_bias=True),
            ],
            name="text_mlp",
        )

        # RoPE Attention
        self.attention = FluxRoPEAttention()

    def build(
        self,
        image_shape,
        text_shape,
        modulation_encoding_shape,
        positional_encoding_shape,
    ):
        """Build every sublayer explicitly.

        This cannot be left to `call`: `compute_output_spec` below stops
        Keras from tracing `call` during functional construction, so nothing
        else would ever create these weights. Symptom if omitted: checkpoint
        conversion dies with "You must build the layer before accessing
        `kernel`" on the JAX and torch backends.
        """
        head_dim = self.hidden_size // self.num_heads
        image_qk_shape = (
            image_shape[0],
            self.num_heads,
            image_shape[1],
            head_dim,
        )
        text_qk_shape = (
            text_shape[0],
            self.num_heads,
            text_shape[1],
            head_dim,
        )

        # Image stream.
        self.image_mod.build(modulation_encoding_shape)
        self.image_norm1.build(image_shape)
        self.image_qkv.build(image_shape)
        self.image_attn_norm.build(image_qk_shape)
        self.image_attn_proj.build(image_shape)
        self.image_norm2.build(image_shape)
        self.image_mlp.build(image_shape)

        # Text stream.
        self.text_mod.build(modulation_encoding_shape)
        self.text_norm1.build(text_shape)
        self.text_qkv.build(text_shape)
        self.text_attn_norm.build(text_qk_shape)
        self.text_attn_proj.build(text_shape)
        self.text_norm2.build(text_shape)
        self.text_mlp.build(text_shape)

    def compute_output_spec(
        self, image, text, modulation_encoding, positional_encoding
    ):
        """Declare the output spec instead of tracing `call`.

        The streams are residual, so shapes are unchanged. Declaring this
        matters for dynamic sequence lengths: otherwise Keras infers the spec
        by running `call` on placeholder tensors, and it picks *independent*
        placeholder lengths for `image`, `text` and `positional_encoding`.
        Those cannot satisfy `len(ids) == len(text) + len(image)`, so RoPE
        broadcasting fails during tracing on the torch and JAX backends.
        """
        return (
            keras.KerasTensor(image.shape, dtype=self.compute_dtype),
            keras.KerasTensor(text.shape, dtype=self.compute_dtype),
        )

    def call(self, image, text, modulation_encoding, positional_encoding):
        """
        Forward pass for the DoubleStreamBlock.

        Args:
            image: Input image tensor.
            text: Input text tensor.
            modulation_encoding: Modulation vector.
            positional_encoding: Positional encoding tensor.

        Returns:
            A `(image, text)` tuple of modified image and text tensors.
        """
        img_mod1, img_mod2 = self.image_mod(modulation_encoding)
        txt_mod1, txt_mod2 = self.text_mod(modulation_encoding)

        img_normed = (
            self.image_norm1(image) * (1 + img_mod1["scale"])
            + img_mod1["shift"]
        )
        txt_normed = (
            self.text_norm1(text) * (1 + txt_mod1["scale"]) + txt_mod1["shift"]
        )

        img_qkv = self.image_qkv(img_normed)
        txt_qkv = self.text_qkv(txt_normed)

        img_q, img_k, img_v = rearrange_symbolic_tensors(
            img_qkv, 3, self.num_heads
        )
        txt_q, txt_k, txt_v = rearrange_symbolic_tensors(
            txt_qkv, 3, self.num_heads
        )

        img_q, img_k = self.image_attn_norm(img_q, img_k)
        txt_q, txt_k = self.text_attn_norm(txt_q, txt_k)

        # NOTE: text comes first. `FluxBackbone` builds the RoPE positions as
        # `concatenate([text_ids, image_ids])`, so the sequence assembled here
        # must use the same order or every token receives the wrong rotary
        # position. This is silent: shapes are identical either way.
        q = ops.concatenate([txt_q, img_q], axis=2)
        k = ops.concatenate([txt_k, img_k], axis=2)
        v = ops.concatenate([txt_v, img_v], axis=2)

        attn_out = self.attention(q, k, v, positional_encoding)

        txt_seq_len = text.shape[1]
        if txt_seq_len is None:
            txt_seq_len = ops.shape(text)[1]
        txt_attn = attn_out[:, :txt_seq_len, :]
        img_attn = attn_out[:, txt_seq_len:, :]

        image = image + img_mod1["gate"] * self.image_attn_proj(img_attn)
        text = text + txt_mod1["gate"] * self.text_attn_proj(txt_attn)

        img_normed_2 = (
            self.image_norm2(image) * (1 + img_mod2["scale"])
            + img_mod2["shift"]
        )
        txt_normed_2 = (
            self.text_norm2(text) * (1 + txt_mod2["scale"]) + txt_mod2["shift"]
        )

        image = image + img_mod2["gate"] * self.image_mlp(img_normed_2)
        text = text + txt_mod2["gate"] * self.text_mlp(txt_normed_2)

        return image, text

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "use_bias": self.use_bias,
            }
        )
        return config


class SingleStreamBlock(keras.Model):
    """
    A DiT block with parallel linear layers.

    As described in https://arxiv.org/abs/2302.05442 and
    adapted for the modulation interface.

    Args:
        hidden_size: int. The hidden dimension size for the model.
        num_heads: int. The number of attention heads.
        mlp_ratio: float, optional. The ratio of the MLP hidden dimension to the
            hidden size. Default is 4.0.
        qk_scale: float, optional. Scaling factor for the query-key product.
            Default is None.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        qk_scale=None,
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
        self.pre_norm = keras.layers.LayerNormalization(
            epsilon=1e-6, scale=False, center=False
        )
        self.modulation = Modulation(hidden_size, double=False)
        self.attention = FluxRoPEAttention()

    def build(
        self, x_shape, modulation_encoding_shape, positional_encoding_shape
    ):
        self.linear1.build(x_shape)
        self.linear2.build(
            (x_shape[0], x_shape[1], self.hidden_size + self.mlp_hidden_dim)
        )

        self.modulation.build(
            modulation_encoding_shape
        )  # Build the modulation layer

        self.norm.build(
            (
                x_shape[0],
                self.num_heads,
                x_shape[1],
                x_shape[-1] // self.num_heads,
            )
        )

    def call(self, x, modulation_encoding, positional_encoding):
        """
        Forward pass for the SingleStreamBlock.

        Args:
            x: KerasTensor. Input tensor.
            modulation_encoding: KerasTensor. Modulation vector.
            positional_encoding: KerasTensor. Positional encoding tensor.

        Returns:
            KerasTensor: The modified input tensor after processing.
        """
        mod, _ = self.modulation(modulation_encoding)
        x_mod = (1 + mod["scale"]) * self.pre_norm(x) + mod["shift"]
        qkv, mlp = ops.split(
            self.linear1(x_mod), [3 * self.hidden_size], axis=-1
        )

        q, k, v = rearrange_symbolic_tensors(qkv, K=3, H=self.num_heads)
        q, k = self.norm(q, k)

        # compute attention
        attn = self.attention(
            q, k=k, v=v, positional_encoding=positional_encoding
        )
        # compute activation in mlp stream, cat again and run second linear
        # layer
        output = self.linear2(
            ops.concatenate(
                (attn, keras.activations.gelu(mlp, approximate=True)), 2
            )
        )
        return x + mod["gate"] * output

    def compute_output_spec(self, x, modulation_encoding, positional_encoding):
        """Declare the output spec instead of tracing `call`.

        See `DoubleStreamBlock.compute_output_spec` — same reasoning; the
        block is residual, so the shape is unchanged.
        """
        return keras.KerasTensor(x.shape, dtype=self.compute_dtype)


class LastLayer(keras.Model):
    """
    Final layer for processing output tensors with adaptive normalization.

    Args:
        hidden_size: int. The hidden dimension size for the model.
        patch_size: int. The size of each patch.
        output_channels: int. The number of output channels.
    """

    def __init__(self, hidden_size, patch_size, output_channels):
        super().__init__()
        self.norm_final = keras.layers.LayerNormalization(
            epsilon=1e-6, scale=False, center=False
        )
        self.linear = keras.layers.Dense(
            patch_size * patch_size * output_channels, use_bias=True
        )
        self.adaLN_modulation = keras.Sequential(
            [
                keras.layers.Activation("silu"),
                keras.layers.Dense(2 * hidden_size, use_bias=True),
            ]
        )

    def call(self, x, modulation_encoding):
        """
        Forward pass for the LastLayer.

        Args:
            x: KerasTensor. Input tensor.
            modulation_encoding: KerasTensor. Modulation vector.

        Returns:
            KerasTensor: The output tensor after final processing.
        """
        shift, scale = ops.split(
            self.adaLN_modulation(modulation_encoding), 2, axis=1
        )
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
