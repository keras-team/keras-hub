import keras
from keras import layers
from keras import ops
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.swin_transformer.swin_transformer_layers import (
    PatchEmbedding,
    SwinTransformerStage,
    PatchMerging
)

def swin_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)

@keras_hub_export("keras_hub.models.SwinTransformerBackbone")
class SwinTransformerBackbone(Backbone):
    """A Swin Transformer backbone network.

    This network implements a hierarchical vision transformer as described in
    ["Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"](https://arxiv.org/abs/2103.14030).
    It includes the patch embedding, transformer stages with shifted windows,
    and final normalization, but not the classification head.

    The default constructor gives a fully customizable, randomly initialized
    Swin Transformer with any number of layers, heads, and embedding dimensions.
    To load preset architectures and weights, use the `from_preset()` constructor.

    Args:
        image_shape: tuple of ints. The shape of the input images, excluding batch dimension.
        patch_size: int. Size of the patches to be extracted from the input images.
        embed_dim: int. Base dimension of the transformer.
        depths: tuple of ints. Number of transformer blocks in each stage.
        num_heads: tuple of ints. Number of attention heads in each stage.
        window_size: int. Size of the attention window.
        mlp_ratio: float. Ratio of mlp hidden dim to embedding dim.
        qkv_bias: bool. If True, add a learnable bias to query, key, value.
        drop: float. Dropout rate.
        attn_drop: float. Dropout rate for attention.
        drop_path: float. Stochastic depth rate.
        patch_norm: bool. If True, add normalization after patch embedding.
        data_format: str. Format of the input data, either "channels_last" or "channels_first".
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights.

    Examples:
    ```python
    # Pretrained Swin Transformer backbone.
    model = keras_hub.models.SwinTransformerBackbone.from_preset(
        "swin_tiny_224"
    )
    model(np.ones((1, 224, 224, 3)))

    # Randomly initialized Swin Transformer with custom config.
    model = keras_hub.models.SwinTransformerBackbone(
        image_shape=(224, 224, 3),
        patch_size=4,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
    )
    model(np.ones((1, 224, 224, 3)))
    ```
    """

    def __init__(
        self,
        image_shape,
        patch_size=4,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.1,
        patch_norm=True,
        data_format="channels_last",
        dtype=None,
        **kwargs,
    ):
        if dtype is None:
            dtype = keras.backend.floatx()

        # === Layers ===
        self.patch_embedding = PatchEmbedding(
            patch_size=patch_size,
            embed_dim=embed_dim,
            norm_layer=layers.LayerNormalization if patch_norm else None,
            data_format=data_format,
            patch_norm=patch_norm,
            name="patch_embedding",
        )

        # Stochastic depth decay rule
        dpr = [float(x) for x in ops.linspace(0.0, drop_path, sum(depths))]

        # === Functional Model ===
        inputs = keras.Input(shape=image_shape)
        x = self.patch_embedding(inputs)
        h, w = image_shape[0] // patch_size, image_shape[1] // patch_size

        # Build stages
        self.stages = []
        for i in range(len(depths)):
            stage = SwinTransformerStage(
                dim=int(embed_dim * 2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                downsample=PatchMerging if (i < len(depths) - 1) else None,
                input_resolution=(h, w),
                name=f"stage_{i}",
            )
            self.stages.append(stage)
            h //= 2
            w //= 2

        # Final norm
        self.norm_layers = [
            layers.LayerNormalization(epsilon=1e-5, name=f"norm_{i}") for i in range(len(depths))
        ]

        # Forward pass
        features = []

        for i, stage in enumerate(self.stages):
            x = stage(x)

            def reshape_and_norm(tensor, norm_layer=self.norm_layers[i]):
                shape = ops.shape(tensor)
                B = shape[0]
                L = shape[1]
                C = shape[2]
                H_float = ops.sqrt(ops.cast(L, x.dtype))
                H = ops.cast(H_float, "int32")
                W = H
                tensor = ops.reshape(tensor, (B, H, W, C))
                return norm_layer(tensor)

            x_reshaped = keras.layers.Lambda(reshape_and_norm)(x)
            features.append(x_reshaped)

        super().__init__(
            inputs=inputs,
            outputs=features[-1],
            dtype=dtype,
            **kwargs
        )

        # === Config ===
        self.image_shape = image_shape
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop = drop
        self.attn_drop = attn_drop
        self.drop_path = drop_path
        self.patch_norm = patch_norm
        self.data_format = data_format

    def get_config(self):
        config = super().get_config()
        config.update({
            "image_shape": self.image_shape,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "depths": self.depths,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "mlp_ratio": self.mlp_ratio,
            "qkv_bias": self.qkv_bias,
            "drop": self.drop,
            "attn_drop": self.attn_drop,
            "drop_path": self.drop_path,
            "patch_norm": self.patch_norm,
            "data_format": self.data_format,
        })
        return config