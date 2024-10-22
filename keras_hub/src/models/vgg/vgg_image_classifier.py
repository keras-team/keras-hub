import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.task import Task
from keras_hub.src.models.vgg.vgg_backbone import VGGBackbone
from keras_hub.src.models.vgg.vgg_image_classifier_preprocessor import (
    VGGImageClassifierPreprocessor,
)


@keras_hub_export("keras_hub.models.VGGImageClassifier")
class VGGImageClassifier(ImageClassifier):
    """VGG image classification task.

    `VGGImageClassifier` tasks wrap a `keras_hub.models.VGGBackbone` and
    a `keras_hub.models.Preprocessor` to create a model that can be used for
    image classification. `VGGImageClassifier` tasks take an additional
    `num_classes` argument, controlling the number of predicted output classes.

    To fine-tune with `fit()`, pass a dataset containing tuples of `(x, y)`
    labels where `x` is a string and `y` is a integer from `[0, num_classes)`.

    Not that unlike `keras_hub.model.ImageClassifier`, the `VGGImageClassifier`
    allows and defaults to `pooling="flatten"`, when inputs are flatten and
    passed through two intermediate dense layers before the final output
    projection.

    Args:
        backbone: A `keras_hub.models.VGGBackbone` instance or a `keras.Model`.
        num_classes: int. The number of classes to predict.
        preprocessor: `None`, a `keras_hub.models.Preprocessor` instance,
            a `keras.Layer` instance, or a callable. If `None` no preprocessing
            will be applied to the inputs.
        pooling: `"flatten"`, `"avg"`, or `"max"`. The type of pooling to apply
            on backbone output. The default is flatten to match the original
            VGG implementation, where backbone inputs will be flattened and
            passed through two dense layers with a `"relu"` activation.
        pooling_hidden_dim: the output feature size of the pooling dense layers.
            This only applies when `pooling="flatten"`.
        activation: `None`, str, or callable. The activation function to use on
            the `Dense` layer. Set `activation=None` to return the output
            logits. Defaults to `"softmax"`.
        head_dtype: `None`, str, or `keras.mixed_precision.DTypePolicy`. The
            dtype to use for the classification head's computations and weights.


    Examples:

    Call `predict()` to run inference.
    ```python
    # Load preset and train
    images = np.random.randint(0, 256, size=(2, 224, 224, 3))
    classifier = keras_hub.models.VGGImageClassifier.from_preset(
        "vgg_16_imagenet"
    )
    classifier.predict(images)
    ```

    Call `fit()` on a single batch.
    ```python
    # Load preset and train
    images = np.random.randint(0, 256, size=(2, 224, 224, 3))
    labels = [0, 3]
    classifier = keras_hub.models.VGGImageClassifier.from_preset(
        "vgg_16_imagenet"
    )
    classifier.fit(x=images, y=labels, batch_size=2)
    ```

    Call `fit()` with custom loss, optimizer and backbone.
    ```python
    classifier = keras_hub.models.VGGImageClassifier.from_preset(
        "vgg_16_imagenet"
    )
    classifier.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(5e-5),
    )
    classifier.backbone.trainable = False
    classifier.fit(x=images, y=labels, batch_size=2)
    ```

    Custom backbone.
    ```python
    images = np.random.randint(0, 256, size=(2, 224, 224, 3))
    labels = [0, 3]
    model = keras_hub.models.VGGBackbone(
        stackwise_num_repeats = [2, 2, 3, 3, 3],
        stackwise_num_filters = [64, 128, 256, 512, 512],
        image_shape = (224, 224, 3),
    )
    classifier = keras_hub.models.VGGImageClassifier(
        backbone=backbone,
        num_classes=4,
    )
    classifier.fit(x=images, y=labels, batch_size=2)
    ```
    """

    backbone_cls = VGGBackbone
    preprocessor_cls = VGGImageClassifierPreprocessor

    def __init__(
        self,
        backbone,
        num_classes,
        preprocessor=None,
        pooling="avg",
        pooling_hidden_dim=4096,
        activation=None,
        dropout=0.0,
        head_dtype=None,
        **kwargs,
    ):
        head_dtype = head_dtype or backbone.dtype_policy
        data_format = getattr(backbone, "data_format", None)

        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor
        if pooling == "avg":
            self.pooler = keras.layers.GlobalAveragePooling2D(
                data_format,
                dtype=head_dtype,
                name="pooler",
            )
        elif pooling == "max":
            self.pooler = keras.layers.GlobalMaxPooling2D(
                data_format,
                dtype=head_dtype,
                name="pooler",
            )
        elif pooling == "flatten":
            self.pooler = keras.Sequential(
                [
                    keras.layers.Flatten(name="flatten"),
                    keras.layers.Dense(pooling_hidden_dim, activation="relu"),
                    keras.layers.Dense(pooling_hidden_dim, activation="relu"),
                ],
                name="pooler",
            )
        else:
            raise ValueError(
                "Unknown `pooling` type. Polling should be either `'avg'` or "
                f"`'max'`. Received: pooling={pooling}."
            )

        self.head = keras.Sequential(
            [
                keras.layers.Conv2D(
                    filters=4096,
                    kernel_size=7,
                    name="fc1",
                    activation=activation,
                    use_bias=True,
                    padding="same",
                ),
                keras.layers.Dropout(
                    rate=dropout,
                    dtype=head_dtype,
                    name="output_dropout",
                ),
                keras.layers.Conv2D(
                    filters=4096,
                    kernel_size=1,
                    name="fc2",
                    activation=activation,
                    use_bias=True,
                    padding="same",
                ),
                self.pooler,
                keras.layers.Dense(
                    num_classes,
                    activation=activation,
                    dtype=head_dtype,
                    name="predictions",
                ),
            ],
            name="head",
        )

        # === Functional Model ===
        inputs = self.backbone.input
        x = self.backbone(inputs)
        outputs = self.head(x)

        # Skip the parent class functional model.
        Task.__init__(
            self,
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

        # === Config ===
        self.num_classes = num_classes
        self.activation = activation
        self.pooling = pooling
        self.pooling_hidden_dim = pooling_hidden_dim
        self.dropout = dropout
        self.preprocessor = preprocessor

    def get_config(self):
        # Backbone serialized in `super`
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "pooling": self.pooling,
                "activation": self.activation,
                "pooling_hidden_dim": self.pooling_hidden_dim,
                "dropout": self.dropout,
            }
        )
        return config
