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

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.models.task import Task


@keras_nlp_export("keras_nlp.models.ImageClassifier")
class ImageClassifier(Task):
    """Base class for all image classification tasks.

    `Classifier` tasks wrap a `keras_nlp.models.Backbone` and
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

    All `Classifier` tasks include a `from_preset()` constructor which can be
    used to load a pre-trained config and weights.
    """

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

    def compile(
        self,
        optimizer="auto",
        loss="auto",
        *,
        metrics="auto",
        **kwargs,
    ):
        """Configures the `ImageClassifier` task for training.

        The `Classifier` task extends the default compilation signature of
        `keras.Model.compile` with defaults for `optimizer`, `loss`, and
        `metrics`. To override these defaults, pass any value
        to these arguments during compilation.

        Args:
            optimizer: `"auto"`, an optimizer name, or a `keras.Optimizer`
                instance. Defaults to `"auto"`, which uses the default optimizer
                for the given model and task. See `keras.Model.compile` and
                `keras.optimizers` for more info on possible `optimizer` values.
            loss: `"auto"`, a loss name, or a `keras.losses.Loss` instance.
                Defaults to `"auto"`, where a
                `keras.losses.SparseCategoricalCrossentropy` loss will be
                applied for the classification task. See
                `keras.Model.compile` and `keras.losses` for more info on
                possible `loss` values.
            metrics: `"auto"`, or a list of metrics to be evaluated by
                the model during training and testing. Defaults to `"auto"`,
                where a `keras.metrics.SparseCategoricalAccuracy` will be
                applied to track the accuracy of the model during training.
                See `keras.Model.compile` and `keras.metrics` for
                more info on possible `metrics` values.
            **kwargs: See `keras.Model.compile` for a full list of arguments
                supported by the compile method.
        """
        if optimizer == "auto":
            optimizer = keras.optimizers.Adam(5e-5)
        if loss == "auto":
            activation = getattr(self, "activation", None)
            activation = keras.activations.get(activation)
            from_logits = activation != keras.activations.softmax
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits)
        if metrics == "auto":
            metrics = [keras.metrics.SparseCategoricalAccuracy()]
        super().compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            **kwargs,
        )