import keras
from keras import layers
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.swin_transformers.swin_transformers_layers import (
    PatchEmbedding,
    SwinTransformerStage,
    PatchMerging
)

@keras_hub_export("keras_hub.models.SwinTransformersBackbone")
class SwinTransformersBackbone(Backbone):
    """Swin Transformer backbone.

    This backbone implements the Swin Transformer architecture as described in
    [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030).
    
    The Swin Transformer is a hierarchical vision transformer that uses shifted 
    windows for self-attention computation. It has several advantages:
    
    1. Hierarchical feature maps with downsampling like CNNs
    2. Linear computational complexity with respect to image size
    3. Support for various vision tasks, including image classification, 
       object detection, and semantic segmentation

    Args:
        image_shape: A tuple or list of 3 integers representing the shape of the
            input image `(height, width, channels)`.
        patch_size: int. The size of each patch (both height and width).
        embed_dim: int. The embedding dimension for the first stage.
        depths: list of ints. Number of transformer blocks in each stage.
        num_heads: list of ints. Number of attention heads in each stage.
        window_size: int. Size of attention window (both height and width).
        mlp_ratio: float. Ratio of MLP hidden dimension to embedding dimension.
        qkv_bias: bool. If True, add a learnable bias to query, key, value.
        dropout_rate: float. Dropout rate for embedding and transformer layers.
        attention_dropout: float. Dropout rate for attention projections.
        path_dropout: float. Stochastic depth rate for transformer blocks.
        patch_norm: bool. If True, add normalization after patch embedding.
        data_format: str. One of `"channels_last"` or `"channels_first"`.
        dtype: The dtype of the layer weights. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to the parent
            `Backbone` class.
    """

    def __init__(
        self,
        image_shape=(224, 224, 3),
        patch_size=4,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        dropout_rate=0.0,
        attention_dropout=0.0,
        path_dropout=0.2,
        patch_norm=True,
        data_format="channels_last",
        dtype=None,
        **kwargs,
    ):
        if len(depths) != len(num_heads):
            raise ValueError(
                f"Length of depths ({len(depths)}) must match "
                f"length of num_heads ({len(num_heads)})"
            )

        self.patch_embedding = PatchEmbedding(
            patch_size=patch_size,
            embed_dim=embed_dim,
            data_format=data_format,
            patch_norm=patch_norm,
            name="patch_embedding"
        )

        self.pos_dropout = layers.Dropout(dropout_rate, name="pos_dropout") if dropout_rate > 0.0 else None

        self.stages = []
        for i, (depth, num_head) in enumerate(zip(depths, num_heads)):
            dim = embed_dim * (2 ** i)
            downsample = PatchMerging(dim=dim // 2, name=f"downsample_{i-1}") if i > 0 else None

            stage = SwinTransformerStage(
                dim=dim,
                depth=depth,
                num_heads=num_head,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout_rate=dropout_rate,
                attention_dropout=attention_dropout,
                path_dropout=path_dropout,
                downsample=downsample,
                name=f"stage_{i}"
            )
            self.stages.append(stage)

        self.norm = layers.LayerNormalization(epsilon=1e-5, name="norm")

        inputs = keras.layers.Input(shape=image_shape)
        x = self.patch_embedding(inputs)
        if self.pos_dropout is not None:
            x = self.pos_dropout(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)

        super().__init__(inputs=inputs, outputs=x, dtype=dtype, **kwargs)

        self.data_format = data_format
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
        self.path_dropout = path_dropout
        self.patch_norm = patch_norm

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
            "dropout_rate": self.dropout_rate,
            "attention_dropout": self.attention_dropout,
            "path_dropout": self.path_dropout,
            "patch_norm": self.patch_norm,
            "data_format": self.data_format,
        })
        return config