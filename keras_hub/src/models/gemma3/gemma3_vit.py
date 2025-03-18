import keras
from keras import ops

from keras_hub.src.models.pali_gemma.pali_gemma_vit import (
    MultiHeadAttentionPooling,
)
from keras_hub.src.models.pali_gemma.pali_gemma_vit import (
    PaliGemmaVitEmbeddings,
)
from keras_hub.src.models.pali_gemma.pali_gemma_vit import (
    PaliGemmaVitEncoderBlock,
)


class Gemma3VitEncoder(keras.layers.Layer):
    def __init__(
        self,
        patch_size,
        image_size,
        hidden_dim,
        num_layers,
        num_heads,
        intermediate_dim,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.encoder_layer_norm = keras.layers.LayerNormalization(
            epsilon=1e-6,
            dtype=dtype,
            name="encoder_layer_norm",
        )
        self.vision_embeddings = PaliGemmaVitEmbeddings(
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            image_size=image_size,
            dtype=dtype,
            name="encoder_embeddings",
        )
        self.resblocks = [
            PaliGemmaVitEncoderBlock(
                self.num_heads,
                self.intermediate_dim,
                dtype=dtype,
                name=f"encoder_block_{i}",
            )
            for i in range(self.num_layers)
        ]

    def build(self, inputs_shape):
        # Collapse batch_size, dummy axis, image_max_length into one.
        inputs_shape = [None] + list(inputs_shape[3:])
        self.vision_embeddings.build(inputs_shape)
        for block in self.resblocks:
            block.build([None, None, self.hidden_dim])
        self.encoder_layer_norm.build([None, None, self.hidden_dim])
        self.built = True

    def call(self, inputs, mask=None):
        inputs_shape = ops.shape(inputs)

        # Collapse batch_size, dummy axis, image_max_length into one.
        inputs = ops.reshape(
            inputs,
            [inputs_shape[0] * inputs_shape[1] * inputs_shape[2]]
            + list(inputs_shape[3:]),
        )

        x = self.vision_embeddings(inputs)
        for block in self.resblocks:
            x = block(x, mask=mask)
        x = self.encoder_layer_norm(x)
        return x

    def compute_output_shape(self, inputs_shape):
        if inputs_shape is None:
            # Fix the compatibility issue with Keras 3.1 where
            # `compute_output_spec` fails to propagate `inputs_shape`
            # correctly, causing it to be `None`.
            inputs_shape = [None, None, None]
        return [
            None,
            (inputs_shape[3] // self.patch_size) ** 2,
            self.hidden_dim,
        ]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "patch_size": self.patch_size,
                "image_size": self.image_size,
            }
        )
        return config


class AveragePooling(keras.layers.Layer):
    def __init__(self, image_size, patch_size, pool_size, **kwargs):
        super().__init__(**kwargs)

        self.width = image_size // patch_size
        # `reduced_width` is the same as `num_vision_tokens_per_image`.
        self.reduced_width = self.width // pool_size

        # Attributes.
        self.image_size = image_size
        self.patch_size = patch_size
        self.pool_size = pool_size

    def build(self, input_shape):
        self.average_pooling = keras.layers.AveragePooling2D(
            pool_size=self.pool_size,
            strides=self.pool_size,
            padding="valid",
            dtype=self.dtype_policy,
            name="average_pooling",
        )

    def call(self, x):
        # reshape `(bsz, height*width, emb_dim)` to
        # `(bsz, width, width, emb_dim)`. `height` should be equal to
        # `width`.
        batch_size, _, hidden_dim = ops.shape(x)
        x = ops.reshape(x, (batch_size, self.width, self.width, hidden_dim))
        x = self.average_pooling(x)
        output = ops.reshape(
            x, (batch_size, self.reduced_width * self.reduced_width, hidden_dim)
        )
        return output

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            self.reduced_width * self.reduced_width,
            input_shape[-1],
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "pool_size": self.pool_size,
            }
        )
        return config


class Gemma3Vit(keras.Model):
    """Vision Transformer (ViT) model for Gemma3.

    Args:
        image_size: int. The height/width of the image. Both height and width is
            expected to be the same.
        image_max_length: int. The maximum number of images per sample (padded).
            Defaults to `None`.
        patch_size: int. The size of each square patch in the input image.
        num_heads: int. The number of attention heads for the vision(image)
            transformer encoder.
        hidden_dim: int. The size of the transformer hidden state at the end
            of each vision transformer layer.
        num_layers: int. The number of transformer layers.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for transformer.
        pooling: string. The encoded vision embeddings are pooled using the
            specified polling setting. The accepted values are `"map"`, `"gap"`,
            `"zero"`, `"average"` or `None`. Defaults to `None`.
        pool_size: int. Used only when `pooling` is `"average"`. Factors by
            which to downscale `(dim1, dim2)`. The same value is used for
            `"strides"`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the models computations and weights. Note that some
            computations, such as softmax and layer normalization will always
            be done a float32 precision regardless of dtype.

    Example:
    ```python
    image = np.random.rand(224, 224, 3)
    vit_model = Gemma3Vit(image_size=224)
    # The output will be of shape:
    # [batch_size, num_vision_tokens_per_image, hidden_dim]
    output = vit_model([image])
    ```
    """

    def __init__(
        self,
        image_size,
        image_max_length,
        patch_size,
        num_heads,
        hidden_dim,
        num_layers,
        intermediate_dim,
        pooling=None,
        pool_size=None,
        dtype=None,
        **kwargs,
    ):
        # === Functional Model ===
        image_input = keras.Input(
            shape=(None, image_max_length, image_size, image_size, 3),
            name="images",
        )
        x = image_input  # Intermediate result.
        x = Gemma3VitEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_dim=intermediate_dim,
            patch_size=patch_size,
            image_size=image_size,
            dtype=dtype,
            name="image_encoder",
        )(x)
        if pooling == "map":
            x = MultiHeadAttentionPooling(
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dtype=dtype,
                name="pooling",
            )(x)
        elif pooling == "gap":
            x = ops.mean(x, axis=1)
        elif pooling == "zero":
            x = x[:, 0]
        elif pooling == "average":
            x = AveragePooling(
                image_size=image_size,
                patch_size=patch_size,
                pool_size=pool_size,
                dtype=dtype,
                name="pooling",
            )(x)

        elif pooling is None:
            x = x
        else:
            raise ValueError(
                "Invalid value for argument `pooling`. "
                "Expected one of 'map', 'gap', 'average', None. "
                f"Received: pooling={pooling}"
            )

        outputs = x
        super().__init__(
            inputs=image_input,
            outputs=outputs,
            **kwargs,
        )

        # === Config ===
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.intermediate_dim = intermediate_dim
        self.pooling = pooling
        self.pool_size = pool_size
        self.image_size = image_size
        self.image_max_length = image_max_length
        self.patch_size = patch_size
        self.num_vision_tokens_per_image = (image_size // patch_size) ** 2
        if pooling == "average":
            self.num_vision_tokens_per_image = (
                self.num_vision_tokens_per_image // (pool_size**2)
            )

        # Before Keras 3.2, there is no `keras.dtype_policies.get`.
        if hasattr(keras.dtype_policies, "get"):
            self.dtype_policy = keras.dtype_policies.get(dtype)
        else:
            if isinstance(dtype, keras.dtype_policies.DTypePolicy):
                dtype = dtype.name
            dtype = dtype or keras.config.dtype_policy().name
            self.dtype_policy = keras.dtype_policies.DTypePolicy(dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "intermediate_dim": self.intermediate_dim,
                "pooling": self.pooling,
                "pool_size": self.pool_size,
                "image_size": self.image_size,
                "image_max_length": self.image_max_length,
                "patch_size": self.patch_size,
            }
        )
        return config
