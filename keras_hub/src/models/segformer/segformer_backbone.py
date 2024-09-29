# Copyright 2024 The KerasHub Authors
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

import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.mix_transformer.mix_transformer_backbone import (
    MiTBackbone,
)
from keras_hub.src.models.segformer.segformer_presets import presets


@keras_hub_export(
    [
        "keras_hub.models.SegFormerBackbone",
        "keras_hub.models.segmentation.SegFormerBackbone",
    ]
)
class SegFormerBackbone(Backbone):
    """A Keras model implementing the SegFormer architecture for semantic
    segmentation.

    References:
        - [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) # noqa: E501
        - [Based on the TensorFlow implementation from DeepVision](https://github.com/DavidLandup0/deepvision/tree/main/deepvision/models/segmentation/segformer) # noqa: E501

    Args:
        backbone: `keras.Model`. The backbone network for the model that is
            used as a feature extractor for the SegFormer encoder.
            It is *intended* to be used only with the MiT backbone model which
            was created specifically for SegFormers. It should either be a
            `keras_cv.models.backbones.backbone.Backbone` or a `tf.keras.Model`
            that implements the `pyramid_level_inputs` property with keys
            "P2", "P3", "P4", and "P5" and layer names as
            values.
        num_classes: int, the number of classes for the detection model,
            including the background class.
        projection_filters: int, number of filters in the
            convolution layer projecting the concatenated features into
            a segmentation map. Defaults to 256`.

    Example:

    Using the class with a custom `backbone`:

    ```python
    import tensorflow as tf
    import keras_hub

    backbone = keras_hub.models.MiTBackbone(
        depths=[2, 2, 2, 2],
        image_shape=(224, 224, 3),
        hidden_dims=[32, 64, 160, 256],
        num_layers=4,
        blockwise_num_heads=[1, 2, 5, 8],
        blockwise_sr_ratios=[8, 4, 2, 1],
        end_value=0.1,
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
    )


    segformer_backbone = keras_hub.models.SegFormerBackbone(backbone=backbone)
    ```
    """

    backbone_cls = MiTBackbone

    def __init__(
        self,
        backbone,
        projection_filters=256,
        **kwargs,
    ):
        if not isinstance(backbone, keras.layers.Layer) or not isinstance(
            backbone, keras.Model
        ):
            raise ValueError(
                "Argument `backbone` must be a `keras.layers.Layer` instance "
                f" or `keras.Model`. Received instead "
                f"backbone={backbone} (of type {type(backbone)})."
            )

        self.feature_extractor = keras.Model(
            backbone.inputs, backbone.pyramid_outputs
        )

        inputs = keras.layers.Input(shape=backbone.input.shape[1:])

        features = self.feature_extractor(inputs)
        # Get H and W of level one output
        _, H, W, _ = features["P1"].shape

        # === Layers ===

        self.mlp_blocks = []

        for feature_dim, feature in zip(backbone.hidden_dims, features):
            self.mlp_blocks.append(
                keras.layers.Dense(
                    projection_filters, name=f"linear_{feature_dim}"
                )
            )

        self.resizing = keras.layers.Resizing(H, W, interpolation="bilinear")
        self.concat = keras.layers.Concatenate(axis=3)
        self.linear_fuse = keras.Sequential(
            [
                keras.layers.Conv2D(
                    filters=projection_filters, kernel_size=1, use_bias=False
                ),
                keras.layers.BatchNormalization(),
                keras.layers.Activation("relu"),
            ]
        )

        # === Functional Model ===

        # Project all multi-level outputs onto the same dimensionality
        # and feature map shape
        multi_layer_outs = []
        for index, (feature_dim, feature) in enumerate(
            zip(backbone.hidden_dims, features)
        ):
            out = self.mlp_blocks[index](features[feature])
            out = self.resizing(out)
            multi_layer_outs.append(out)

        # Concat now-equal feature maps
        concatenated_outs = self.concat(multi_layer_outs[::-1])

        # Fuse concatenated features into a segmentation map
        seg = self.linear_fuse(concatenated_outs)

        super().__init__(
            inputs=inputs,
            outputs=seg,
            **kwargs,
        )

        self.projection_filters = projection_filters
        self.backbone = backbone

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "projection_filters": self.projection_filters,
                "backbone": keras.saving.serialize_keras_object(self.backbone),
            }
        )
        return config
