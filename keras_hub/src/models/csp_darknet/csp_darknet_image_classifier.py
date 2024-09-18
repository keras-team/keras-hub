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
import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.csp_darknet.csp_darknet_backbone import (
    CSPDarkNetBackbone,
)
from keras_hub.src.models.image_classifier import ImageClassifier


@keras_hub_export("keras_hub.models.CSPDarkNetImageClassifier")
class CSPDarkNetImageClassifier(ImageClassifier):
    """CSPDarkNet image classifier task model.

    Args:
        backbone: A `keras_hub.models.CSPDarkNetBackbone` instance.
        num_classes: int. The number of classes to predict.
        activation: `None`, str or callable. The activation function to use on
            the `Dense` layer. Set `activation=None` to return the output
            logits. Defaults to `"softmax"`.

    To fine-tune with `fit()`, pass a dataset containing tuples of `(x, y)`
    where `x` is a tensor and `y` is a integer from `[0, num_classes)`.
    All `ImageClassifier` tasks include a `from_preset()` constructor which can
    be used to load a pre-trained config and weights.

    Examples:

    Call `predict()` to run inference.
    ```python
    # Load preset and train
    images = np.ones((2, 224, 224, 3), dtype="float32")
    classifier = keras_hub.models.CSPDarkNetImageClassifier.from_preset(
        "csp_darknet_tiny_imagenet")
    classifier.predict(images)
    ```

    Call `fit()` on a single batch.
    ```python
    # Load preset and train
    images = np.ones((2, 224, 224, 3), dtype="float32")
    labels = [0, 3]
    classifier = keras_hub.models.CSPDarkNetImageClassifier.from_preset(
        "csp_darknet_tiny_imagenet")
    classifier.fit(x=images, y=labels, batch_size=2)
    ```

    Call `fit()` with custom loss, optimizer and backbone.
    ```python
    classifier = keras_hub.models.CSPDarkNetImageClassifier.from_preset(
        "csp_darknet_tiny_imagenet")
    classifier.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(5e-5),
    )
    classifier.backbone.trainable = False
    classifier.fit(x=images, y=labels, batch_size=2)
    ```

    Custom backbone.
    ```python
    images = np.ones((2, 224, 224, 3), dtype="float32")
    labels = [0, 3]
    backbone = keras_hub.models.CSPDarkNetBackbone(
        stackwise_num_filters=[128, 256, 512, 1024],
        stackwise_depth=[3, 9, 9, 3],
        include_rescaling=False,
        block_type="basic_block",
        image_shape = (224, 224, 3),
    )
    classifier = keras_hub.models.CSPDarkNetImageClassifier(
        backbone=backbone,
        num_classes=4,
    )
    classifier.fit(x=images, y=labels, batch_size=2)
    ```
    """

    backbone_cls = CSPDarkNetBackbone

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
