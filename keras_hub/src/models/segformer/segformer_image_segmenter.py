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
from keras_hub.src.models.image_segmenter import ImageSegmenter
from keras_hub.src.models.segformer.segformer_backbone import SegFormerBackbone


@keras_hub_export(
    [
        "keras_hub.models.SegFormerImageSegmenter",
        "keras_hub.models.segmentation.SegFormerImageSegmenter",
    ]
)
class SegFormerImageSegmenter(ImageSegmenter):
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

    Using the class with a `backbone`:

    ```python
    import tensorflow as tf
    import keras_cv

    images = np.ones(shape=(1, 96, 96, 3))
    labels = np.zeros(shape=(1, 96, 96, 1))

    encoder = keras_hub.models.MiTBackbone(
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

    segformer_backbone = keras_hub.models.SegFormerBackbone(backbone=encoder)
    segformer = SegFormerImageSegmenter(backbone=segformer_backbone, num_classes=4)

    # Evaluate model
    model(images)

    # Train model
    model.compile(
        optimizer="adam",
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    model.fit(images, labels, epochs=3)
    ```
    """

    backbone_cls = SegFormerBackbone

    def __init__(
        self,
        backbone,
        num_classes,
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

        inputs = backbone.input

        # === Layers ===
        self.backbone = backbone
        self.dropout = keras.layers.Dropout(0.1)
        self.output_segmentation = keras.layers.Conv2D(
            filters=num_classes, kernel_size=1, activation="softmax"
        )
        self.resizing = keras.layers.Resizing(
            height=inputs.shape[1],
            width=inputs.shape[2],
            interpolation="bilinear",
        )

        # === Functional Model ===
        x = self.backbone(inputs)
        x = self.dropout(x)
        x = self.output_segmentation(x)
        output = self.resizing(x)

        super().__init__(
            inputs=inputs,
            outputs=output,
            **kwargs,
        )

        # === Config ===
        self.num_classes = num_classes
        self.projection_filters = projection_filters
        self.backbone = backbone

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "projection_filters": self.projection_filters,
                "backbone": keras.saving.serialize_keras_object(self.backbone),
            }
        )
        return config
