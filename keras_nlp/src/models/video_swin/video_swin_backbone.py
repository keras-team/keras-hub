# Copyright 2024 The KerasNLP Authors
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
from functools import partial

import keras
import numpy as np
from keras import layers

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.models.backbone import Backbone
from keras_nlp.src.models.video_swin.video_swin_layers import (
    VideoSwinBasicLayer,
)
from keras_nlp.src.models.video_swin.video_swin_layers import (
    VideoSwinPatchingAndEmbedding,
)
from keras_nlp.src.models.video_swin.video_swin_layers import (
    VideoSwinPatchMerging,
)


@keras_nlp_export("keras_nlp.models.VideoSwinBackbone")
class VideoSwinBackbone(Backbone):
    """A Video Swin Transformer backbone model.

    Args:
        image_shape (tuple[int], optional): The size of the input video in
            `(depth, height, width, channel)` format.
            Defaults to `(32, 224, 224, 3)`.
        include_rescaling (bool, optional): Whether to rescale the inputs. If
            set to `True`, inputs will be passed through a `Rescaling(1/255.0)` layer
            and normalize with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
            Defaults to `False`.
        patch_size (int | tuple(int)): The patch size for depth, height, and width
            dimensions respectively. Default: (2,4,4).
        embed_dim (int): Number of linear projection output channels.
            Default to 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
            Default to [2, 2, 6, 2]
        num_heads (tuple[int]): Number of attention head of each stage.
            Default to [3, 6, 12, 24]
        window_size (int): The window size for depth, height, and width
            dimensions respectively. Default to [8, 7, 7].
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            Default to 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
            Default to True.
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
            Default to None.
        drop_rate (float): Float between 0 and 1. Fraction of the input units to drop.
            Default: 0.
        attn_drop_rate (float): Float between 0 and 1. Attention dropout rate.
            Default: 0.
        drop_path_rate (float): Float between 0 and 1. Stochastic depth rate.
            Default: 0.2.
        patch_norm (bool): If True, add layer normalization after patch embedding.
            Default to False.

    Example:
    ```python
    # Build video swin backbone without top layer
    model = VideoSwinSBackbone(
        include_rescaling=True, image_shape=(8, 256, 256, 3),
    )
    videos = keras.ops.ones((1, 8, 256, 256, 3))
    outputs = model.predict(videos)
    ```

    References:
        - [Video Swin Transformer](https://arxiv.org/abs/2106.13230)
        - [Official Code](https://github.com/SwinTransformer/Video-Swin-Transformer)
    """

    def __init__(
        self,
        include_rescaling=False,
        image_shape=(32, 224, 224, 3),
        embed_dim=96,
        patch_size=[2, 4, 4],
        window_size=[8, 7, 7],
        mlp_ratio=4.0,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        qkv_bias=True,
        qk_scale=None,
        **kwargs,
    ):

        # === Functional Model ===

        inputs = keras.layers.Input(shape=image_shape)
        x = inputs

        if include_rescaling:
            x = layers.Rescaling(1.0 / 255.0)(x)

            x = layers.Normalization(
                mean=[0.485, 0.456, 0.406],
                variance=[0.229**2, 0.224**2, 0.225**2],
                name="normalization",
            )(x)

        norm_layer = partial(layers.LayerNormalization, epsilon=1e-05)

        x = VideoSwinPatchingAndEmbedding(
            patch_size=patch_size,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
            name="videoswin_patching_and_embedding",
        )(x)
        x = layers.Dropout(drop_rate, name="pos_drop")(x)

        dpr = np.linspace(0.0, drop_path_rate, sum(depths)).tolist()
        num_layers = len(depths)
        for i in range(num_layers):
            layer = VideoSwinBasicLayer(
                input_dim=int(embed_dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                norm_layer=norm_layer,
                downsampling_layer=(
                    VideoSwinPatchMerging if (i < num_layers - 1) else None
                ),
                name=f"videoswin_basic_layer_{i + 1}",
            )
            x = layer(x)

        x = norm_layer(axis=-1, epsilon=1e-05, name="videoswin_top_norm")(x)
        super().__init__(inputs=inputs, outputs=x, **kwargs)

        # === Config ===
        self.include_rescaling = include_rescaling
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.patch_norm = patch_norm
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.num_layers = len(depths)
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.depths = depths
        self.image_shape = image_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "include_rescaling": self.include_rescaling,
                "image_shape": self.image_shape,
                "embed_dim": self.embed_dim,
                "patch_norm": self.patch_norm,
                "window_size": self.window_size,
                "patch_size": self.patch_size,
                "mlp_ratio": self.mlp_ratio,
                "drop_rate": self.drop_rate,
                "drop_path_rate": self.drop_path_rate,
                "attn_drop_rate": self.attn_drop_rate,
                "depths": self.depths,
                "num_heads": self.num_heads,
                "qkv_bias": self.qkv_bias,
                "qk_scale": self.qk_scale,
            }
        )
        return config
