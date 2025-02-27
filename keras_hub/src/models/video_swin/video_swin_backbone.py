# Copyright 2024 The Kerashub Authors
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
import copy
from functools import partial

import numpy as np
from keras import layers

from keras_hub.src.api_export import keras_hub_export
# from keras_hub.src.backend import keras
import keras
from keras_hub.src.models import utils
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.video_swin.video_swin_backbone_presets import (  # noqa: E501
    backbone_presets,
)
from keras_hub.src.models.video_swin.video_swin_backbone_presets import (  # noqa: E501
    backbone_presets_with_weights,
)
from keras_hub.src.models.video_swin.video_swin_layers import (
    VideoSwinBasicLayer,
)
from keras_hub.src.models.video_swin.video_swin_layers import (
    VideoSwinPatchingAndEmbedding,
)
from keras_hub.src.models.video_swin.video_swin_layers import (
    VideoSwinPatchMerging,
)
from keras_hub.src.utils.python_utils import classproperty


@keras_hub_export("keras_hub_export.models.VideoSwinBackbone", package="keras_hub_export.models")
class VideoSwinBackbone(Backbone):
    """A Video Swin Transformer backbone model.
    References:
        - [Video Swin Transformer](https://arxiv.org/abs/2106.13230)
        - [Official Code](https://github.com/SwinTransformer/Video-Swin-Transformer)

    Args:
        input_shape : tuple[int]. The size of the input video in
            `(depth, height, width, channel)` format.
            Defaults to `(32, 224, 224, 3)`.
        patch_size : tuple(int). The patch size for depth, height, and width
            dimensions respectively. Default: (2,4,4).
        embed_dim : int. Number of linear projection output channels.
            Default to 96.
        depth :tuple[int]. Depth of each Swin Transformer stage.
            Default to [2, 2, 6, 2]
        num_heads : tuple[int]. Number of attention head of each stage.
            Default to [3, 6, 12, 24]
        window_size : int. The window size for depth, height, and width
            dimensions respectively. Default to [8, 7, 7].
        mlp_ratio : float. Ratio of mlp hidden dim to embedding dim.
            Default to 4.
        qkv_bias : bool. If True, add a learnable bias to query, key, value.
            Default to True.
        qk_scale : float. Override default qk scale of head_dim ** -0.5 if set.
            Default to None.
        dropout_rate : float. Float between 0 and 1. Fraction of the input units to drop.
            Default: 0.
        attn_dropout_rate : float. Float between 0 and 1. Attention dropout rate.
            Default: 0.
        drop_path_rate : float. floatFloat between 0 and 1. Stochastic depth rate.
            Default: 0.2.
        patch_norm : bool. If True, add layer normalization after patch embedding.
            Default to False.

    Example:
    ```python
    # Build video swin backbone without top layer
   from keras_hub.src.models.video_swin.video_swin_layers import VideoSwinBasicLayer
    model = VideoSwinBasicLayer(
         input_shape=(8, 256, 256, 3),
    )
    videos = keras.ops.ones((1, 8, 256, 256, 3))
    outputs = model.predict(videos)
    ```
    """  # noqa: E501

    def __init__(
        self,
        *,
        input_shape=(32, 224, 224, 3),
        input_tensor=None,
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
        # Parse input specification.
        input_spec = utils.parse_model_inputs(
            input_shape, input_tensor, name="videos"
        )

        # Check that the input video is well specified.
        if (
            input_spec.shape[-4] is None
            or input_spec.shape[-3] is None
            or input_spec.shape[-2] is None
        ):
            raise ValueError(
                "Depth, height and width of the video must be specified"
                " in `input_shape`."
            )

        x = input_spec

        # if include_rescaling:
        #     # Use common rescaling strategy across keras_cv
        #     x = keras.layers.Rescaling(1.0 / 255.0)(x)

        #     # VideoSwin scales inputs based on the ImageNet mean/stddev.
        #     # Officially, Videw Swin takes tensor of [0-255] ranges.
        #     # And use mean=[123.675, 116.28, 103.53] and
        #     # std=[58.395, 57.12, 57.375] for normalization.
        #     # So, if include_rescaling is set to True, then, to match with the
        #     # official scores, following normalization should be added.
        #     x = layers.Normalization(
        #         mean=[0.485, 0.456, 0.406],
        #         variance=[0.229**2, 0.224**2, 0.225**2],
        #     )(x)

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
        super().__init__(inputs=input_spec, outputs=x, **kwargs)

        # self.include_rescaling = include_rescaling
        self.input_tensor = input_tensor
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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
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

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return copy.deepcopy(backbone_presets)

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return copy.deepcopy(backbone_presets_with_weights)

    @property
    def pyramid_level_inputs(self):
        raise NotImplementedError(
            "The `VideoSwinBackbone` model doesn't compute"
            " pyramid level features."
        )
