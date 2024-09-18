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
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.resnet.resnet_backbone import ResNetBackbone
from keras_hub.src.models.resnet.resnet_image_classifier_preprocessor import (
    ResNetImageClassifierPreprocessor,
)


@keras_hub_export("keras_hub.models.ResNetImageClassifier")
class ResNetImageClassifier(ImageClassifier):
    """ResNet image classifier task model.

    Args:
        backbone: A `keras_hub.models.ResNetBackbone` instance.
        num_classes: int. The number of classes to predict.
        activation: `None`, str or callable. The activation function to use on
            the `Dense` layer. Set `activation=None` to return the output
            logits. Defaults to `"softmax"`.
        head_dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The
            dtype to use for the classification head's computations and weights.

    To fine-tune with `fit()`, pass a dataset containing tuples of `(x, y)`
    where `x` is a tensor and `y` is a integer from `[0, num_classes)`.
    All `ImageClassifier` tasks include a `from_preset()` constructor which can
    be used to load a pre-trained config and weights.

    Examples:

    Call `predict()` to run inference.
    ```python
    # Load preset and train
    images = np.ones((2, 224, 224, 3), dtype="float32")
    classifier = keras_hub.models.ResNetImageClassifier.from_preset("resnet50")
    classifier.predict(images)
    ```

    Call `fit()` on a single batch.
    ```python
    # Load preset and train
    images = np.ones((2, 224, 224, 3), dtype="float32")
    labels = [0, 3]
    classifier = keras_hub.models.ResNetImageClassifier.from_preset("resnet50")
    classifier.fit(x=images, y=labels, batch_size=2)
    ```

    Call `fit()` with custom loss, optimizer and backbone.
    ```python
    classifier = keras_hub.models.ResNetImageClassifier.from_preset("resnet50")
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
    backbone = keras_hub.models.ResNetBackbone(
        stackwise_num_filters=[64, 64, 64],
        stackwise_num_blocks=[2, 2, 2],
        stackwise_num_strides=[1, 2, 2],
        block_type="basic_block",
        use_pre_activation=True,
        include_rescaling=False,
        pooling="avg",
    )
    classifier = keras_hub.models.ResNetImageClassifier(
        backbone=backbone,
        num_classes=4,
    )
    classifier.fit(x=images, y=labels, batch_size=2)
    ```
    """

    backbone_cls = ResNetBackbone
    preprocessor_cls = ResNetImageClassifierPreprocessor

    def __init__(
        self,
        backbone,
        num_classes,
        preprocessor=None,
        pooling="avg",
        activation=None,
        head_dtype=None,
        **kwargs,
    ):
        head_dtype = head_dtype or backbone.dtype_policy

        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor
        if pooling == "avg":
            self.pooler = keras.layers.GlobalAveragePooling2D(
                data_format=backbone.data_format, dtype=head_dtype
            )
        elif pooling == "max":
            self.pooler = keras.layers.GlobalAveragePooling2D(
                data_format=backbone.data_format, dtype=head_dtype
            )
        else:
            raise ValueError(
                "Unknown `pooling` type. Polling should be either `'avg'` or "
                f"`'max'`. Received: pooling={pooling}."
            )
        self.output_dense = keras.layers.Dense(
            num_classes,
            activation=activation,
            dtype=head_dtype,
            name="predictions",
        )

        # === Functional Model ===
        inputs = self.backbone.input
        x = self.backbone(inputs)
        x = self.pooler(x)
        outputs = self.output_dense(x)
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

        # === Config ===
        self.num_classes = num_classes
        self.activation = activation
        self.pooling = pooling

    def get_config(self):
        # Backbone serialized in `super`
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "pooling": self.pooling,
                "activation": self.activation,
            }
        )
        return config
