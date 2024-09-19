# Copyright 2023 The KerasHub Authors
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

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.vgg.vgg_backbone import VGGBackbone


@keras_hub_export("keras_hub.models.VGGImageClassifier")
class VGGImageClassifier(ImageClassifier):
    """VGG16 image classifier task model.

    Args:
      backbone: A `keras_hub.models.VGGBackbone` instance.
      num_classes: int, number of classes to predict.
      pooling: str, type of pooling layer. Must be one of "avg", "max".
      activation: Optional `str` or callable, defaults to "softmax". The
          activation function to use on the Dense layer. Set `activation=None`
          to return the output logits.

    To fine-tune with `fit()`, pass a dataset containing tuples of `(x, y)`
    labels where `x` is a string and `y` is a integer from `[0, num_classes)`.
    All `ImageClassifier` tasks include a `from_preset()` constructor which can be
    used to load a pre-trained config and weights.

    Examples:
    Train from preset
    ```python
    # Load preset and train
    images = np.ones((2, 224, 224, 3), dtype="float32")
    labels = [0, 3]
    classifier = keras_hub.models.VGGImageClassifier.from_preset(
        'vgg_16_image_classifier')
    classifier.fit(x=images, y=labels, batch_size=2)

    # Re-compile (e.g., with a new learning rate).
    classifier.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(5e-5),
        jit_compile=True,
    )

    # Access backbone programmatically (e.g., to change `trainable`).
    classifier.backbone.trainable = False
    # Fit again.
    classifier.fit(x=images, y=labels, batch_size=2)
    ```
    Custom backbone
    ```python
    images = np.ones((2, 224, 224, 3), dtype="float32")
    labels = [0, 3]

    backbone = keras_hub.models.VGGBackbone(
        stackwise_num_repeats = [2, 2, 3, 3, 3],
        stackwise_num_filters = [64, 128, 256, 512, 512],
        image_shape = (224, 224, 3),
        include_rescaling = False,
        pooling = "avg",
    )
    classifier = keras_hub.models.VGGImageClassifier(
        backbone=backbone,
        num_classes=4,
    )
    classifier.fit(x=images, y=labels, batch_size=2)
    ```
    """

    backbone_cls = VGGBackbone

    def __init__(
        self,
        backbone,
        num_classes,
        activation="softmax",
        preprocessor=None,  # adding this dummy arg for saved model test
        # TODO: once preprocessor flow is figured out, this needs to be updated
        **kwargs,
    ):
        # === Layers ===
        self.backbone = backbone
        self.output_dense = keras.layers.Dense(
            num_classes,
            activation=activation,
            name="predictions",
        )

        # === Functional Model ===
        inputs = self.backbone.input
        x = self.backbone(inputs)
        outputs = self.output_dense(x)

        # Instantiate using Functional API Model constructor
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
