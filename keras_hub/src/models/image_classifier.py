import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.task import Task


@keras_hub_export("keras_hub.models.ImageClassifier")
class ImageClassifier(Task):
    """Base class for all image classification tasks.

    `ImageClassifier` tasks wrap a `keras_hub.models.Backbone` and
    a `keras_hub.models.Preprocessor` to create a model that can be used for
    image classification. `ImageClassifier` tasks take an additional
    `num_classes` argument, controlling the number of predicted output classes.

    To fine-tune with `fit()`, pass a dataset containing tuples of `(x, y)`
    labels where `x` is a string and `y` is a integer from `[0, num_classes)`.
    All `ImageClassifier` tasks include a `from_preset()` constructor which can
    be used to load a pre-trained config and weights.

    Args:
        backbone: A `keras_hub.models.Backbone` instance or a `keras.Model`.
        num_classes: int. The number of classes to predict.
        preprocessor: `None`, a `keras_hub.models.Preprocessor` instance,
            a `keras.Layer` instance, or a callable. If `None` no preprocessing
            will be applied to the inputs.
        pooling: `"avg"` or `"max"`. The type of pooling to apply on backbone
            output. Defaults to average pooling.
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
    classifier = keras_hub.models.ImageClassifier.from_preset(
        "resnet_50_imagenet"
    )
    classifier.predict(images)
    ```

    Call `fit()` on a single batch.
    ```python
    # Load preset and train
    images = np.random.randint(0, 256, size=(2, 224, 224, 3))
    labels = [0, 3]
    classifier = keras_hub.models.ImageClassifier.from_preset(
        "resnet_50_imagenet"
    )
    classifier.fit(x=images, y=labels, batch_size=2)
    ```

    Call `fit()` with custom loss, optimizer and backbone.
    ```python
    classifier = keras_hub.models.ImageClassifier.from_preset(
        "resnet_50_imagenet"
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
    backbone = keras_hub.models.ResNetBackbone(
        stackwise_num_filters=[64, 64, 64],
        stackwise_num_blocks=[2, 2, 2],
        stackwise_num_strides=[1, 2, 2],
        block_type="basic_block",
        use_pre_activation=True,
        pooling="avg",
    )
    classifier = keras_hub.models.ImageClassifier(
        backbone=backbone,
        num_classes=4,
    )
    classifier.fit(x=images, y=labels, batch_size=2)
    ```
    """

    def __init__(
        self,
        backbone,
        num_classes,
        preprocessor=None,
        pooling="avg",
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
        else:
            raise ValueError(
                "Unknown `pooling` type. Polling should be either `'avg'` or "
                f"`'max'`. Received: pooling={pooling}."
            )
        self.output_dropout = keras.layers.Dropout(
            dropout,
            dtype=head_dtype,
            name="output_dropout",
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
        x = self.output_dropout(x)
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
        self.dropout = dropout

    def get_config(self):
        # Backbone serialized in `super`
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "pooling": self.pooling,
                "activation": self.activation,
                "dropout": self.dropout,
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

        The `ImageClassifier` task extends the default compilation signature of
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
