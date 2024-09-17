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

import keras

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.models.deeplab_v3.deeplab_v3_backbone import (
    DeepLabV3Backbone,
)
from keras_nlp.src.models.image_segmenter import ImageSegmenter


@keras_nlp_export("keras_nlp.models.DeepLabV3ImageSegmenter")
class DeepLabV3ImageSegmenter(ImageSegmenter):
    """DeepLabV3 and DeeplabV3 and DeeplabV3Plus segmentation task.

    Args:
        backbone: A `keras_nlp.models.DeepLabV3` instance.
        num_classes: int, the number of classes for the detection model. Note
            that the `num_classes` contains the background class, and the
            classes from the data should be represented by integers with range
            [0, `num_classes`].
        activation: str or callable. The activation function to use on
            the `Dense` layer. Set `activation=None` to return the output
            logits. Defaults to `None`.

    Example:
    ```python
    images = np.ones(shape=(1, 96, 96, 3))
    labels = np.zeros(shape=(1, 96, 96, 1))
    feature_pyramid_model = keras_nlp.models.DeepLabV3Backbone.from_preset("deeplabv3_resnet50")

    model = keras_hub.models.DeepLabV3ImageSegmenter(
        num_classes=3,
        projection_filters=48,
        low_level_feature_key="P2",
        spatial_pyramid_pooling_key="P5",
    )

    # Evaluate model
    model(images)

    # Train model
    model.compile(
        optimizer="adam",
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(images, labels, epochs=3)
    ```
    """

    backbone_cls = DeepLabV3Backbone
    preprocessor_cls = None

    def __init__(
        self,
        backbone,
        num_classes,
        activation=None,
        preprocessor=None,
        **kwargs,
    ):
        data_format = keras.config.image_data_format()
        # === Layers ===
        self.backbone = backbone
        self.output_conv = keras.layers.Conv2D(
            name="segmentation_output",
            filters=num_classes,
            kernel_size=1,
            use_bias=False,
            padding="same",
            activation=activation,
            data_format=data_format,
            # Force the dtype of the classification layer to float32
            # to avoid the NAN loss issue when used with mixed
            # precision API.
            dtype="float32",
        )

        # === Functional Model ===
        inputs = self.backbone.input
        x = self.backbone(inputs)
        outputs = self.output_conv(x)
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

        # === Config ===
        self.num_classes = num_classes
        self.activation = activation

    def get_config(self):
        # Backbone serialized in `super`
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "activation": self.activation,
            }
        )
        return config
