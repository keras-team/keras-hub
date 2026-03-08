import keras
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.swin_transformer.swin_transformer_layers import (
    PatchEmbedding,
)
from keras_hub.src.models.swin_transformer.swin_transformer_layers import (
    PatchMerging,
)
from keras_hub.src.models.swin_transformer.swin_transformer_layers import (
    SwinTransformerStage,
)



@keras_hub_export("keras_hub.models.SwinTransformerBackbone")
class SwinTransformerBackbone(Backbone):
    """A Swin Transformer backbone network.

    This network implements a hierarchical vision transformer as described in
    ["Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"](https://arxiv.org/abs/2103.14030).
    It includes the patch embedding, transformer stages with shifted windows,
    and final normalization, but not the classification head.

    The default constructor gives a fully customizable, randomly initialized
    Swin Transformer with any number of layers, heads, and embedding dimensions.
    To load preset architectures and weights, use the `from_preset()`
    constructor.

    Args:
        image_shape: tuple of ints. The shape of the input images, excluding
            batch dimension.
        embed_dim: int. Base dimension of the transformer.
        depths: tuple of ints. Number of transformer blocks in each stage.
        num_heads: tuple of ints. Number of attention heads in each stage.
        window_size: int. Size of the attention window.
        patch_size: int. Size of the patches to be extracted from the input
            images. Defaults to `4`.
        mlp_ratio: float. Ratio of mlp hidden dim to embedding dim.
            Defaults to `4.0`.
        qkv_bias: bool. If True, add a learnable bias to query, key, value.
            Defaults to `True`.
        dropout_rate: float. Dropout rate. Defaults to `0.0`.
        attention_dropout: float. Dropout rate for attention.
            Defaults to `0.0`.
        drop_path: float. Stochastic depth rate. Defaults to `0.1`.
        patch_norm: bool. If True, add normalization after patch embedding.
            Defaults to `True`.
        data_format: str. Format of the input data, either `"channels_last"`
            or `"channels_first"`. Defaults to `None` (which uses
            `"channels_last"`).
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights. Defaults to `None`.

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
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
    )
    model(np.ones((1, 224, 224, 3)))
    ```
    """

    def __init__(
        self,
        image_shape,
        embed_dim,
        depths,
        num_heads,
        window_size,
        patch_size=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        dropout_rate=0.0,
        attention_dropout=0.0,
        drop_path=0.1,
        patch_norm=True,
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        if data_format is None:
            data_format = "channels_last"

        dpr = [float(x) for x in ops.linspace(0.0, drop_path, sum(depths))]

        # === Layers ===
        inputs = keras.Input(shape=image_shape)

        patch_embedding_layer = PatchEmbedding(
            patch_size=patch_size,
            embed_dim=embed_dim,
            norm_layer=(layers.LayerNormalization if patch_norm else None),
            data_format=data_format,
            patch_norm=patch_norm,
            dtype=dtype,
            name="patch_embedding",
        )
        x = patch_embedding_layer(inputs)
        x = layers.Dropout(dropout_rate, dtype=dtype, name="pos_drop")(x)

        h = image_shape[0] // patch_size
        w = image_shape[1] // patch_size

        stage_layers = []
        for i in range(len(depths)):
            stage = SwinTransformerStage(
                dim=int(embed_dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout_rate=dropout_rate,
                attention_dropout=attention_dropout,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                downsample=(PatchMerging if (i < len(depths) - 1) else None),
                input_resolution=(h, w),
                dtype=dtype,
                name=f"stage_{i}",
            )
            x = stage(x)
            stage_layers.append(stage)
            if i < len(depths) - 1:
                h //= 2
                w //= 2

        norm_layer = layers.LayerNormalization(
            epsilon=1e-5, dtype=dtype, name="norm"
        )
        x = norm_layer(x)

        super().__init__(inputs=inputs, outputs=x, dtype=dtype, **kwargs)

        # === Layer references ===
        self.patch_embedding = patch_embedding_layer
        self.stages = stage_layers
        self.norm = norm_layer

        # === Config ===
        self.image_shape = image_shape
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.drop_path = drop_path
        self.patch_norm = patch_norm
        self.data_format = data_format

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_shape": self.image_shape,
                "patch_size": self.patch_size,
                "embed_dim": self.embed_dim,
                "depths": self.depths,
                "num_heads": self.num_heads,
                "window_size": self.window_size,
                "mlp_ratio": self.mlp_ratio,
                "qkv_bias": self.qkv_bias,
                "dropout_rate": self.dropout_rate,
                "attention_dropout": self.attention_dropout,
                "drop_path": self.drop_path,
                "patch_norm": self.patch_norm,
                "data_format": self.data_format,
            }
        )
        return config
