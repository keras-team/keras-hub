import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.densenet.densenet_backbone import DenseNetBackbone
from keras_hub.src.models.densenet.densenet_image_classifier_preprocessor import (
    DenseNetImageClassifierPreprocessor,
)
from keras_hub.src.models.image_classifier import ImageClassifier


@keras_hub_export("keras_hub.models.DenseNetImageClassifier")
class DenseNetImageClassifier(ImageClassifier):
    """DenseNet image classifier task model.

    To fine-tune with `fit()`, pass a dataset containing tuples of `(x, y)`
    where `x` is a tensor and `y` is a integer from `[0, num_classes)`.
    All `ImageClassifier` tasks include a `from_preset()` constructor which can
    be used to load a pre-trained config and weights.

    Args:
        backbone: A `keras_hub.models.DenseNetBackbone` instance.
        num_classes: int. The number of classes to predict.
        activation: `None`, str or callable. The activation function to use on
            the `Dense` layer. Set `activation=None` to return the output
            logits. Defaults to `None`.
        pooling: A pooling layer to use before the final classification layer,
            must be one of "avg" or "max". Use "avg" for
            `GlobalAveragePooling2D` and "max" for "GlobalMaxPooling2D.
        preprocessor: A `keras_hub.models.DenseNetImageClassifierPreprocessor`
            or `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model.

    Examples:

    Call `predict()` to run inference.
    ```python
    # Load preset and train
    images = np.ones((2, 224, 224, 3), dtype="float32")
    classifier = keras_hub.models.DenseNetImageClassifier.from_preset(
        "densenet121_imagenet")
    classifier.predict(images)
    ```

    Call `fit()` on a single batch.
    ```python
    # Load preset and train
    images = np.ones((2, 224, 224, 3), dtype="float32")
    labels = [0, 3]
    classifier = keras_hub.models.DenseNetImageClassifier.from_preset(
        "densenet121_imagenet")
    classifier.fit(x=images, y=labels, batch_size=2)
    ```

    Call `fit()` with custom loss, optimizer and backbone.
    ```python
    classifier = keras_hub.models.DenseNetImageClassifier.from_preset(
        "densenet121_imagenet")
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
    backbone = keras_hub.models.DenseNetBackbone(
        stackwise_num_filters=[128, 256, 512, 1024],
        stackwise_depth=[3, 9, 9, 3],
        block_type="basic_block",
        image_shape = (224, 224, 3),
    )
    classifier = keras_hub.models.DenseNetImageClassifier(
        backbone=backbone,
        num_classes=4,
    )
    classifier.fit(x=images, y=labels, batch_size=2)
    ```
    """

    backbone_cls = DenseNetBackbone
    preprocessor_cls = DenseNetImageClassifierPreprocessor

    def __init__(
        self,
        backbone,
        num_classes,
        activation=None,
        pooling="avg",
        preprocessor=None,
        **kwargs,
    ):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor
        if pooling == "avg":
            self.pooler = keras.layers.GlobalAveragePooling2D()
        elif pooling == "max":
            self.pooler = keras.layers.GlobalMaxPooling2D()
        else:
            raise ValueError(
                "Unknown `pooling` type. Polling should be either `'avg'` or "
                f"`'max'`. Received: pooling={pooling}."
            )
        self.output_dense = keras.layers.Dense(
            num_classes,
            activation=activation,
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
                "activation": self.activation,
                "pooling": self.pooling,
            }
        )
        return config
