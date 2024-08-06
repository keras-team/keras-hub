# Copyright 2023 The KerasNLP Authors
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

from keras_nlp.src.models.image_classifier import ImageClassifier
from keras_nlp.src.models.vgg16 import VGG16Backbone


class VGG16ImageClassifier(ImageClassifier):
    """Base class for all image classification tasks.

    `ImageClassifier` tasks wrap a `keras_nlp.models.Backbone` and
    a `keras_nlp.models.Preprocessor` to create a model that can be used for
    image classification.

    Args:
      backbone: `keras.Model` instance, the backbone architecture of the
          classifier called on the inputs. Pooling will be called on the last
          dimension of the backbone output.
      num_classes: int, number of classes to predict.
      pooling: str, type of pooling layer. Must be one of "avg", "max".
      activation: Optional `str` or callable, defaults to "softmax". The
          activation function to use on the Dense layer. Set `activation=None`
          to return the output logits.

    To fine-tune with `fit()`, pass a dataset containing tuples of `(x, y)`
    labels where `x` is a string and `y` is a integer from `[0, num_classes)`.

    All `ImageClassifier` tasks include a `from_preset()` constructor which can be
    used to load a pre-trained config and weights.
    """

    backbone_cls = VGG16Backbone

    def __init__(
        self,
        backbone,
        num_classes,
        pooling="avg",
        activation="softmax",
        **kwargs,
    ):
        # === Layers ===
        if pooling == "avg":
            pooling_layer = keras.layers.GlobalAveragePooling2D(name="avg_pool")
        elif pooling == "max":
            pooling_layer = keras.layers.GlobalMaxPooling2D(name="max_pool")
        else:
            raise ValueError(
                f'`pooling` must be one of "avg", "max". Received: {pooling}.'
            )
        # === Functional Model ===
        inputs = backbone.input
        x = backbone(inputs)
        x = pooling_layer(x)
        outputs = keras.layers.Dense(
            num_classes,
            activation=activation,
            name="predictions",
        )(x)

        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )
        # === Config ===
        self.backbone = backbone
        self.num_classes = num_classes
        self.pooling = pooling
        self.activation = activation

    def get_config(self):
        # Backbone serialized in `super`
        config = super().get_config()
        config.update(
            {
                "backbone": keras.layers.serialize(self.backbone),
                "num_classes": self.num_classes,
                "pooling": self.pooling,
                "activation": self.activation,
            }
        )
        return config
