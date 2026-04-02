import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.gemma4.gemma4_decoder_block import (
    Gemma4VisionDecoderBlock,
)
from keras_hub.src.models.gemma4.gemma4_layers import Gemma4VNorm


@keras_hub_export("keras_hub.models.Gemma4VisionEncoder")
class Gemma4VisionEncoder(keras.Model):
    """Vision Transformer (ViT) encoder for Gemma4.

    This encoder is architecturally different from the Gemma3 vision encoder.
    Rather than using a separate CLIP-style ViT, Gemma4 uses the same
    transformer block style as the text decoder (with 4 norms per block,
    Q/K/V normalization) with bidirectional (non-causal) attention.

    Position information is encoded via two separate learnable position-
    embedding tables — one for the x-axis and one for the y-axis — whose
    outputs are added to the patch features. This 2D decomposed embedding
    can represent any image height and width independently.

    After encoding, the patch sequence is spatially pooled down to a fixed
    `output_dim`-wide representation and then projected into the text hidden
    dimension.

    Args:
        image_size: int. The height/width of the (square) image. Must be
            divisible by `patch_size * pool_size`.
        patch_size: int. Size of each square patch in pixels.
        num_heads: int. Number of attention heads in each vision transformer
            layer.
        hidden_dim: int. Hidden dimension of the vision transformer blocks.
        num_layers: int. Number of transformer layers in the vision encoder.
        intermediate_dim: int. Intermediate FFW dimension in each block.
        head_dim: int. Dimension of each attention head.
        output_dim: int. Dimension to project encoded patches to (should equal
            the text backbone's `hidden_dim`).
        num_key_value_heads: int. For grouped-query attention. Defaults to
            `num_heads` (MHA).
        pool_size: int. Spatial pooling factor applied after the transformer.
            The output sequence length equals
            `(image_size // patch_size // pool_size) ** 2`. Must evenly divide
            `image_size // patch_size`. Defaults to `3`.
        position_embedding_size: int. Number of learnable entries in each
            (x or y) position embedding table. Should be at least
            `image_size // patch_size`. Defaults to `1024`.
        layer_norm_epsilon: float. Epsilon for layer normalisations. Defaults
            to `1e-6`.
        dropout: float. Dropout probability. Defaults to `0`.
        dtype: Compute dtype. Defaults to `None` (uses Keras global policy).

    Example:
    ```python
    import numpy as np

    vision_encoder = keras_hub.models.Gemma4VisionEncoder(
        image_size=768,
        patch_size=16,
        num_heads=12,
        hidden_dim=768,
        num_layers=12,
        intermediate_dim=3072,
        head_dim=64,
        output_dim=2304,
        pool_size=3,
    )
    # pixel_values: (batch, num_images, num_patches, patch_dim)
    # For a 768x768 image with patch_size=16: num_patches=48*48=2304,
    # patch_dim=16*16*3=768.
    pixel_values = np.ones((1, 1, 2304, 768), dtype="float32")
    # pixel_position_ids: (batch, num_images, num_patches, 2)
    pixel_position_ids = np.zeros((1, 1, 2304, 2), dtype="int32")
    output = vision_encoder(
        {"pixel_values": pixel_values, "pixel_position_ids": pixel_position_ids}
    )
    # output.shape == (1, 1, 256, 2304)
    # (batch, num_images, pooled_patches, output_dim)
    ```
    """

    def __init__(
        self,
        image_size,
        patch_size,
        num_heads,
        hidden_dim,
        num_layers,
        intermediate_dim,
        head_dim,
        output_dim,
        num_key_value_heads=None,
        pool_size=3,
        position_embedding_size=1024,
        rope_max_wavelength=100.0,
        layer_norm_epsilon=1e-6,
        dropout=0,
        use_clipped_linears=True,
        standardize=False,
        dtype=None,
        **kwargs,
    ):
        # Vision encoder always runs in float32 to match other frameworks.
        # Normalize any DTypePolicy object or mixed-precision string to float32.
        if hasattr(dtype, "variable_dtype"):
            dtype = "float32"
        elif dtype is not None and dtype != "float32":
            dtype = "float32"

        if num_key_value_heads is None:
            num_key_value_heads = num_heads

        # === Functional Model ===
        pixel_values_input = keras.Input(
            shape=(None, None, 3 * patch_size * patch_size),
            name="pixel_values",
        )
        pixel_position_ids_input = keras.Input(
            shape=(None, None, 2),
            dtype="int32",
            name="pixel_position_ids",
        )

        x = Gemma4VisionEncoderBlock(
            image_size=image_size,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            intermediate_dim=intermediate_dim,
            position_embedding_size=position_embedding_size,
            rope_max_wavelength=rope_max_wavelength,
            layer_norm_epsilon=layer_norm_epsilon,
            dropout=dropout,
            use_clipped_linears=use_clipped_linears,
            dtype=dtype,
            name="image_encoder",
        )(pixel_values_input, pixel_position_ids=pixel_position_ids_input)

        x = Gemma4VisionAveragePooling(
            image_size=image_size,
            patch_size=patch_size,
            pool_size=pool_size,
            dtype=dtype,
            name="pooling",
        )(x, pixel_position_ids=pixel_position_ids_input)

        # Project to text hidden dimension.
        x = Gemma4VisionOutput(
            output_dim=output_dim,
            layer_norm_epsilon=layer_norm_epsilon,
            standardize=standardize,
            dtype=dtype,
            name="vision_output_encoder",
        )(x)

        outputs = x
        super().__init__(
            inputs={
                "pixel_values": pixel_values_input,
                "pixel_position_ids": pixel_position_ids_input,
            },
            outputs=outputs,
            **kwargs,
        )

        # === Config ===
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.intermediate_dim = intermediate_dim
        self.head_dim = head_dim
        self.output_dim = output_dim
        self.num_key_value_heads = num_key_value_heads
        self.pool_size = pool_size
        self.position_embedding_size = position_embedding_size
        self.rope_max_wavelength = rope_max_wavelength
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout
        self.num_vision_tokens_per_image = (
            (image_size // patch_size) // pool_size
        ) ** 2

        self.use_clipped_linears = use_clipped_linears
        self.standardize = standardize

        if hasattr(keras.dtype_policies, "get"):
            self.dtype_policy = keras.dtype_policies.get(dtype)
        else:
            if isinstance(dtype, keras.dtype_policies.DTypePolicy):
                dtype = dtype.name
            dtype = dtype or keras.config.dtype_policy().name
            self.dtype_policy = keras.dtype_policies.DTypePolicy(dtype)

    def get_config(self):
        config = {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "intermediate_dim": self.intermediate_dim,
            "head_dim": self.head_dim,
            "output_dim": self.output_dim,
            "num_key_value_heads": self.num_key_value_heads,
            "pool_size": self.pool_size,
            "position_embedding_size": self.position_embedding_size,
            "rope_max_wavelength": self.rope_max_wavelength,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "dropout": self.dropout,
            "use_clipped_linears": self.use_clipped_linears,
            "standardize": self.standardize,
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Gemma4VisionPatchEmbedder(keras.layers.Layer):
    """Converts raw image pixels to patch embeddings with 2D position info.

    Patches are projected to `hidden_dim` via a flat linear layer. Position
    information is encoded via two separate learnable embedding tables (one
    for x, one for y), whose outputs are added elementwise to the patch
    features.

    Args:
        image_size: int. Height/width of the input image.
        patch_size: int. Size of each square patch in pixels.
        hidden_dim: int. Dimensionality of the output embeddings.
        position_embedding_size: int. Number of entries in each axis' position
            embedding table. Must be ≥ `image_size // patch_size`.
        dtype: Compute/storage dtype for layer weights.
    """

    def __init__(
        self,
        image_size,
        patch_size,
        hidden_dim,
        position_embedding_size=1024,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.position_embedding_size = position_embedding_size

        self.num_patches_per_side = image_size // patch_size
        self.num_patches = self.num_patches_per_side**2

        # Project flattened patches to hidden_dim.
        self.input_proj = keras.layers.Dense(
            units=hidden_dim,
            use_bias=False,
            dtype=dtype,
            name="input_proj",
        )

        # Use the variable dtype for add_weight (not the full policy name).
        weight_dtype = getattr(dtype, "variable_dtype", dtype)
        self.position_embedding_table = self.add_weight(
            name="position_embedding_table",
            shape=(2, position_embedding_size, hidden_dim),
            initializer="ones",
            trainable=True,
            dtype=weight_dtype,
        )

    def build(self, input_shape):
        self.input_proj.build(
            (None, self.num_patches, 3 * self.patch_size * self.patch_size)
        )
        self.built = True

    def call(self, pixel_values, pixel_position_ids):
        """Embed a batch of images.

        Args:
            pixel_values: float tensor of shape
                `(batch, num_patches, patch_pixels)`.
            pixel_position_ids: int tensor of shape `(batch, num_patches, 2)`.

        Returns:
            float tensor of shape `(batch, num_patches, hidden_dim)`.
        """
        # Gemma4 applies no normalization and instead scales in model code
        # Inputs are in [0, 1], scale to [-1, 1] to match HF.
        pixel_values = 2.0 * (pixel_values - 0.5)
        x = self.input_proj(pixel_values)

        # Clamp padding patches (which are -1) to 0 so they don't break one_hot.
        clamped_positions = ops.maximum(pixel_position_ids, 0)

        # Create one-hot for positions
        one_hot = ops.one_hot(
            clamped_positions, num_classes=self.position_embedding_size
        )  # (B, T, 2, P)

        # Equation: btip (B, T, 2, P) and iph (2, P, H) -> bth (B, T, H)
        pos_embeds = ops.einsum(
            "btip,iph->bth", one_hot, self.position_embedding_table
        )

        # Zero out position embeddings for padding patches (-1)
        is_padding = ops.all(
            ops.equal(pixel_position_ids, -1), axis=-1, keepdims=True
        )
        pos_embeds = pos_embeds * (1.0 - ops.cast(is_padding, pos_embeds.dtype))
        x = x + pos_embeds
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.hidden_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "hidden_dim": self.hidden_dim,
                "position_embedding_size": self.position_embedding_size,
            }
        )
        return config


class Gemma4VisionEncoderBlock(keras.layers.Layer):
    """Stack of Gemma4 decoder blocks used as the vision encoder.

    All layers use full (bidirectional) attention — there is no causal masking
    or sliding-window restriction for vision.

    Args:
        image_size: int. Height/width of the input image.
        patch_size: int. Size of each square patch in pixels.
        hidden_dim: int. Hidden dimension of each transformer layer.
        num_layers: int. Number of transformer layers.
        num_heads: int. Number of query attention heads.
        num_key_value_heads: int. Number of key/value heads (GQA).
        head_dim: int. Per-head dimension.
        intermediate_dim: int. FFW intermediate dimension.
        position_embedding_size: int. Size of each position embedding table.
        layer_norm_epsilon: float. Epsilon for layer normalisations.
        dropout: float. Dropout probability.
        dtype: Compute/storage dtype.
    """

    def __init__(
        self,
        image_size,
        patch_size,
        hidden_dim,
        num_layers,
        num_heads,
        num_key_value_heads,
        head_dim,
        intermediate_dim,
        position_embedding_size=10240,
        rope_max_wavelength=100.0,
        layer_norm_epsilon=1e-6,
        dropout=0,
        use_clipped_linears=True,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_dim = intermediate_dim
        self.position_embedding_size = position_embedding_size
        self.rope_max_wavelength = rope_max_wavelength
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout
        self.use_clipped_linears = use_clipped_linears

        self.patch_embedder = Gemma4VisionPatchEmbedder(
            image_size=image_size,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            position_embedding_size=position_embedding_size,
            dtype=dtype,
            name="patch_embedder",
        )

        # All vision layers use full (non-causal, bidirectional) attention.
        self.encoder_blocks = [
            Gemma4VisionDecoderBlock(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                head_dim=head_dim,
                num_query_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                layer_norm_epsilon=layer_norm_epsilon,
                rope_wavelength=rope_max_wavelength,
                dropout=dropout,
                use_clipped_linears=use_clipped_linears,
                dtype=dtype,
                name=f"encoder_block_{i}",
            )
            for i in range(num_layers)
        ]

    def build(self, inputs_shape):
        if isinstance(inputs_shape, dict) and "pixel_values" in inputs_shape:
            pixel_values_shape = inputs_shape["pixel_values"]
        else:
            pixel_values_shape = inputs_shape
        single_patch_shape = [None, pixel_values_shape[-1]]
        self.patch_embedder.build(single_patch_shape)
        patch_out_shape = (
            None,
            pixel_values_shape[1],
            self.hidden_dim,
        )
        for block in self.encoder_blocks:
            block.build(patch_out_shape)
        self.built = True

    def call(self, pixel_values, pixel_position_ids=None):
        inputs_shape = ops.shape(pixel_values)
        # Collapse the (batch, num_images) leading dims into one.
        flat_pv = ops.reshape(
            pixel_values,
            [inputs_shape[0] * inputs_shape[1]] + list(inputs_shape[2:]),
        )

        if pixel_position_ids is not None:
            pos_shape = ops.shape(pixel_position_ids)
            flat_pos = ops.reshape(
                pixel_position_ids,
                [pos_shape[0] * pos_shape[1]] + list(pos_shape[2:]),
            )
        else:
            flat_pos = None

        x = self.patch_embedder(flat_pv, flat_pos)

        # Precompute padding mask: True where position_ids == (-1, -1).
        # After each encoder block, zero padding patch residuals so they stay
        # silent in subsequent blocks. HF achieves this via its bidirectional
        # attention mask (padding queries get all-inf rows → zero attention
        # output), but blocking full rows causes NaN in KH's softmax. Zeroing
        # the residual explicitly is numerically identical for real patches
        # (their attention never depends on padding KV, which is masked) and
        # ensures padding patches don't accumulate garbage across layers.
        if flat_pos is not None:
            is_padding = ops.all(
                ops.equal(flat_pos, -1), axis=-1, keepdims=True
            )  # (B*I, N, 1) bool
            real_mask = ops.cast(~is_padding, x.dtype)
        else:
            real_mask = None

        for i, block in enumerate(self.encoder_blocks):
            x, _ = block(x, position_ids=flat_pos)
            # Zero padding patches after each block — they stay zero entering
            # the next block's attention so they cannot contaminate real KV.
            if real_mask is not None:
                x = x * real_mask

        # Un-flatten the images dimension.
        flat_shape = ops.shape(x)
        x = ops.reshape(
            x,
            [inputs_shape[0], inputs_shape[1], flat_shape[1], flat_shape[2]],
        )
        return x

    def compute_output_shape(self, inputs_shape):
        if inputs_shape is None:
            return (None, None, None, self.hidden_dim)
        if isinstance(inputs_shape, dict) and "pixel_values" in inputs_shape:
            pixel_values_shape = inputs_shape["pixel_values"]
        else:
            pixel_values_shape = inputs_shape
        # pixel_values_shape is (batch, num_images, num_patches, patch_dim)
        return (
            pixel_values_shape[0],
            pixel_values_shape[1],
            pixel_values_shape[2],
            self.hidden_dim,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
                "intermediate_dim": self.intermediate_dim,
                "position_embedding_size": self.position_embedding_size,
                "rope_max_wavelength": self.rope_max_wavelength,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "use_clipped_linears": self.use_clipped_linears,
            }
        )
        return config


class Gemma4VisionAveragePooling(keras.layers.Layer):
    """Spatial average-pooling of patch embeddings.

    Pools the input tokens by averaging patches within a `k^2` grid
    using their positions.

    Args:
        image_size: int. Ignored in new dynamic pooling but kept for
            signature compatibility.
        patch_size: int. Size of each square patch.
        pool_size: int. The spatial pooling factor.
    """

    def __init__(self, image_size, patch_size, pool_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.pool_size = pool_size

    def build(self, input_shape):
        pass

    def call(self, inputs, pixel_position_ids):
        shape = ops.shape(inputs)
        if len(shape) == 4:
            batch_size, max_images, num_patches, hidden_dim = shape
            if max_images == 0 or num_patches == 0:
                return inputs
            inputs = ops.reshape(
                inputs, (batch_size * max_images, num_patches, hidden_dim)
            )
            pixel_position_ids = ops.reshape(
                pixel_position_ids, (batch_size * max_images, num_patches, 2)
            )
        else:
            batch_size, num_patches, hidden_dim = shape
            if num_patches == 0:
                return inputs
            max_images = 1

        k = self.pool_size
        k_squared = k**2

        clamped_positions = ops.maximum(pixel_position_ids, 0)

        output_length = num_patches // k_squared

        kernel_x = clamped_positions[..., 0] // k
        kernel_y = clamped_positions[..., 1] // k

        # Derive the pooled-grid width from the maximum x-position in the
        # batch. For a rectangular image (e.g. 57 patch-cols × 42 patch-rows
        # pooled by k=3) this gives n_w_pooled=19, which `round(sqrt(280))`
        # would get wrong (17). The `ops.max` call is dynamic but is only
        # used for indexing, not for slicing, so JAX JIT accepts it.
        n_w_pooled = (ops.max(clamped_positions[..., 0]) + 1) // k
        kernel_idxs = kernel_x + n_w_pooled * kernel_y

        # Zero out padding patches (position_ids == -1 for both coords) before
        # pooling, matching HF's masked_fill in Gemma4VisionPooler.forward.
        # Without this, padding patches clamped to bin 0 contaminate the
        # top-left pool bin with their non-zero encoder outputs.
        is_padding = ops.all(
            ops.equal(pixel_position_ids, -1), axis=-1, keepdims=True
        )
        inputs = inputs * (1.0 - ops.cast(is_padding, inputs.dtype))

        # Use total output length for num_classes to prevent
        # dropping valid patches!
        weights = ops.one_hot(kernel_idxs, num_classes=output_length)
        weights = ops.cast(weights, self.compute_dtype) / ops.cast(
            k_squared, self.compute_dtype
        )

        output = ops.matmul(ops.transpose(weights, (0, 2, 1)), inputs)

        # Apply root_hidden_size scaling, matching HF!
        hidden_dim = ops.shape(inputs)[-1]
        root_hidden_size = ops.cast(hidden_dim**0.5, self.compute_dtype)
        output = output * root_hidden_size

        if len(shape) == 4:
            output = ops.reshape(
                output, (batch_size, max_images, output_length, hidden_dim)
            )
        else:
            pass  # shape: (batch_size * max_images, output_length, hidden_dim)

        return output

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 4:
            return (input_shape[0], input_shape[1], None, input_shape[-1])
        return (input_shape[0], None, input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "pool_size": self.pool_size,
            }
        )
        return config


class Gemma4VisionOutput(keras.layers.Layer):
    """Post-pooling projection of vision embeddings into the text space.

    In Gemma4, after spatial pooling and sqrt-scaling, the pooled embeddings
    are projected from `vision_hidden_dim` to `text_hidden_dim` via a linear
    layer followed by a pure-L2 (parameter-free) RMS normalisation.

    Args:
        output_dim: int. Output dimension (text backbone hidden dimension).
        layer_norm_epsilon: float. Epsilon for the output RMS norm.
    """

    def __init__(
        self, output_dim, layer_norm_epsilon=1e-6, standardize=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.layer_norm_epsilon = layer_norm_epsilon
        self.standardize = standardize

    def build(self, input_shape):
        hidden_dim = input_shape[-1]

        if self.standardize:
            self.std_bias = self.add_weight(
                name="std_bias",
                shape=(hidden_dim,),
                initializer="zeros",
                trainable=False,
            )
            self.std_scale = self.add_weight(
                name="std_scale",
                shape=(hidden_dim,),
                initializer="ones",
                trainable=False,
            )

        # Pure RMS norm (no learnable scale) — matches
        # Gemma4MultimodalEmbedder (with_scale=False) in HF.
        self.output_norm = Gemma4VNorm(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="output_norm",
        )
        # Since Norm doesn't change shape, input_shape to Dense is the same!
        self.output_norm.build(input_shape)

        self.vision_input_projection = keras.layers.Dense(
            units=self.output_dim,
            use_bias=False,
            dtype=self.dtype_policy,
            name="vision_input_projection",
        )
        self.vision_input_projection.build(input_shape)

    def call(self, inputs):
        x = inputs
        if self.standardize:
            x = (x - self.std_bias) * self.std_scale
        x = self.output_norm(x)
        x = self.vision_input_projection(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.output_dim,)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "standardize": self.standardize,
            }
        )
        return config
