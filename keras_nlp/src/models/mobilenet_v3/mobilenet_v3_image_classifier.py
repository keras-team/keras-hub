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
from keras_nlp.src.models.image_classifier import ImageClassifier
from keras_nlp.src.models.mobilenet_v3.mobilenet_v3_backbone import (
    MobileNetV3Backbone,
)


@keras_nlp_export("keras_nlp.models.MobileNetV3ImageClassifier")
class MobileNetV3ImageClassifier(ImageClassifier):
    """MobileNetV3 image classifier task model.

    To fine-tune with `fit()`, pass a dataset containing tuples of `(x, y)`
    where `x` is a tensor and `y` is a integer from `[0, num_classes)`.
    All `ImageClassifier` tasks include a `from_preset()` constructor which can
    be used to load a pre-trained config and weights.

    Args:
        backbone: A `keras_nlp.models.MobileNetV3Backbone` instance.
        num_classes: int. The number of classes to predict.
        activation: `None`, str or callable. The activation function to use on
            the `Dense` layer. Set `activation=None` to return the output
            logits. Defaults to `"softmax"`.

    Examples:

    Call `predict()` to run inference.
    ```python
    # Load preset and train
    images = np.ones((2, 224, 224, 3), dtype="float32")
    classifier = keras_nlp.models.MobileNetV3ImageClassifier.from_preset(
        "mobilenet_v3_small_imagenet")
    classifier.predict(images)
    ```

    Custom backbone.
    ```python
    images = np.ones((2, 224, 224, 3), dtype="float32")
    labels = [0, 3]
    model = MobileNetV3Backbone(
        stackwise_expansion=[1, 72.0 / 16, 88.0 / 24, 4, 6, 6, 3, 3, 6, 6, 6],
        stackwise_filters=[16, 24, 24, 40, 40, 40, 48, 48, 96, 96, 96],
        stackwise_kernel_size=[3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5],
        stackwise_stride=[2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1],
        stackwise_se_ratio=[0.25, None, None, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
        stackwise_activation=["relu", "relu", "relu", "hard_swish", "hard_swish", "hard_swish", "hard_swish", "hard_swish", "hard_swish", "hard_swish", "hard_swish"],
        include_rescaling=False,
    )
    classifier = keras_nlp.models.MobileNetV3ImageClassifier(
        backbone=backbone,
        num_classes=4,
    )
    classifier.fit(x=images, y=labels, batch_size=2)
    ```
    """

    backbone_cls = MobileNetV3Backbone

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
