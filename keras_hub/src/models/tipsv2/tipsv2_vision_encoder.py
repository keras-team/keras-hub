"""TIPSv2 Vision Encoder.

A DINOv2-style Vision Transformer that produces spatially rich features
aligned with text embeddings. Outputs CLS token, register tokens, and
per-patch spatial tokens.
"""

import math

import keras
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.tipsv2.tipsv2_layers import TIPSv2PatchEmbedding
from keras_hub.src.models.tipsv2.tipsv2_layers import TIPSv2VisionBlock
from keras_hub.src.utils.keras_utils import standardize_data_format


class TIPSv2VisionEmbedding(keras.layers.Layer):
    """Embedding layer for TIPSv2 vision encoder.

    Converts images to patch embeddings, prepends CLS token, adds
    positional embeddings, and inserts register tokens.

    Args:
        hidden_dim: int. Token embedding dimension.
        patch_size: int. Size of each square patch.
        image_size: int. Expected input image size (height=width).
        num_register_tokens: int. Number of register tokens.
        data_format: str. Data format for conv.
    """

    def __init__(
        self,
        hidden_dim,
        patch_size,
        image_size,
        num_register_tokens=1,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_register_tokens = num_register_tokens
        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embedding = TIPSv2PatchEmbedding(
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            image_size=image_size,
            data_format=data_format,
            dtype=self.dtype_policy,
            name="patch_embedding",
        )

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.hidden_dim),
            initializer="zeros",
            trainable=True,
        )
        self.position_embeddings = self.add_weight(
            name="position_embeddings",
            shape=(1, self.num_patches + 1, self.hidden_dim),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )
        if self.num_register_tokens > 0:
            self.register_tokens = self.add_weight(
                name="register_tokens",
                shape=(1, self.num_register_tokens, self.hidden_dim),
                initializer=keras.initializers.TruncatedNormal(stddev=1e-6),
                trainable=True,
            )
        else:
            self.register_tokens = None
        self.patch_embedding.build(input_shape)
        self.built = True

    def _interpolate_pos_encoding(self, x):
        """Interpolate position embeddings to match actual input."""
        pos_embed = ops.cast(self.position_embeddings, dtype=x.dtype)
        cls_pos = pos_embed[:, :1]
        patch_pos = pos_embed[:, 1:]

        dim = self.hidden_dim
        w0 = self.image_size // self.patch_size
        h0 = self.image_size // self.patch_size
        num_patches_dim = int(math.sqrt(self.num_patches))

        patch_pos = ops.reshape(
            patch_pos, (1, num_patches_dim, num_patches_dim, dim)
        )
        patch_pos = ops.image.resize(
            patch_pos, size=(h0, w0), interpolation="bilinear"
        )
        patch_pos = ops.reshape(patch_pos, (1, -1, dim))
        return ops.concatenate([cls_pos, patch_pos], axis=1)

    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]

        # Patch embed.
        x = self.patch_embedding(inputs)  # (B, N, D)

        # Prepend CLS token.
        cls_tokens = ops.broadcast_to(
            self.cls_token, (batch_size, 1, self.hidden_dim)
        )
        x = ops.concatenate([cls_tokens, x], axis=1)  # (B, N+1, D)

        # Add position embeddings.
        x = x + self._interpolate_pos_encoding(x)

        # Insert register tokens after CLS.
        if self.register_tokens is not None:
            reg = ops.broadcast_to(
                self.register_tokens,
                (batch_size, self.num_register_tokens, self.hidden_dim),
            )
            x = ops.concatenate(
                [x[:, :1], reg, x[:, 1:]], axis=1
            )  # (B, 1+R+N, D)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "patch_size": self.patch_size,
                "image_size": self.image_size,
                "num_register_tokens": self.num_register_tokens,
            }
        )
        return config


@keras_hub_export("keras_hub.models.TIPSv2VisionEncoder")
class TIPSv2VisionEncoder(Backbone):
    """TIPSv2 vision encoder based on DINOv2 ViT architecture.

    This encoder processes images through patch embedding, adds positional
    embeddings (with interpolation support for arbitrary resolutions),
    prepends a CLS token and register tokens, then applies a stack of
    transformer blocks. The output provides CLS, register, and patch tokens.

    The default constructor gives a fully customizable, randomly initialized
    model. To load preset architectures and weights, use `from_preset`.

    Args:
        patch_size: int. Size of each square patch.
        hidden_dim: int. Transformer hidden dimension.
        num_layers: int. Number of transformer blocks.
        num_heads: int. Number of attention heads.
        mlp_ratio: float. Ratio of MLP hidden dim to hidden_dim.
            Defaults to `4.0`.
        init_values: float. LayerScale init value. Defaults to `1.0`.
        num_register_tokens: int. Number of register tokens.
            Defaults to `1`.
        ffn_layer: str. FFN type, "mlp" or "swiglu". Defaults to `"mlp"`.
        image_shape: tuple. Input shape (H, W, C). Defaults to
            `(448, 448, 3)`.
        data_format: str. Image data format. Defaults to `None`.
        dtype: str or Policy. Dtype for computations. Defaults to `None`.

    Example:
    ```python
    import numpy as np
    import keras_hub

    encoder = keras_hub.models.TIPSv2VisionEncoder(
        patch_size=14,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        image_shape=(448, 448, 3),
    )
    images = np.random.rand(1, 448, 448, 3).astype("float32")
    outputs = encoder({"images": images})
    ```
    """

    def __init__(
        self,
        patch_size,
        hidden_dim,
        num_layers,
        num_heads,
        mlp_ratio=4.0,
        init_values=1.0,
        num_register_tokens=1,
        ffn_layer="mlp",
        image_shape=(448, 448, 3),
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        data_format = standardize_data_format(data_format)
        if data_format == "channels_last":
            height = image_shape[0]
        else:
            height = image_shape[1]

        # === Layers ===
        self.embeddings = TIPSv2VisionEmbedding(
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            image_size=height,
            num_register_tokens=num_register_tokens,
            data_format=data_format,
            dtype=dtype,
            name="embeddings",
        )
        self.vision_blocks = []
        for i in range(num_layers):
            block = TIPSv2VisionBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                init_values=init_values,
                ffn_layer=ffn_layer,
                dtype=dtype,
                name=f"block_{i}",
            )
            self.vision_blocks.append(block)
        self.layernorm = layers.LayerNormalization(
            epsilon=1e-6, dtype=dtype, name="layernorm"
        )

        # === Functional Model ===
        image_input = layers.Input(shape=image_shape, name="images")

        x = self.embeddings(image_input)

        # Transformer blocks.
        for block in self.vision_blocks:
            x = block(x)

        x = self.layernorm(x)

        # Split outputs.
        cls_out = x[:, :1]  # (B, 1, D)
        if num_register_tokens > 0:
            reg_out = x[:, 1 : 1 + num_register_tokens]  # (B, R, D)
            patch_out = x[:, 1 + num_register_tokens :]  # (B, N, D)
        else:
            reg_out = x[:, :0]  # (B, 0, D) - empty
            patch_out = x[:, 1:]

        super().__init__(
            inputs={"images": image_input},
            outputs={
                "cls_token": cls_out,
                "register_tokens": reg_out,
                "patch_tokens": patch_out,
            },
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.init_values = init_values
        self.num_register_tokens = num_register_tokens
        self.ffn_layer = ffn_layer
        self.image_shape = image_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "init_values": self.init_values,
                "num_register_tokens": self.num_register_tokens,
                "ffn_layer": self.ffn_layer,
                "image_shape": self.image_shape,
            }
        )
        return config
