import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.deit.deit_backbone import DeiTBackbone
from keras_hub.src.models.deit.deit_image_classifier_preprocessor import (
    DeiTImageClassifierPreprocessor,
)
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.task import Task


@keras_hub_export("keras_hub.models.DeiTImageClassifier")
class DeiTImageClassifier(ImageClassifier):
    """DeiT image classification task.

    `DeiTImageClassifier` tasks wrap a `keras_hub.models.DeiTBackbone` and
    a `keras_hub.models.Preprocessor` to create a model that can be used for
    image classification. `DeiTImageClassifier` tasks take an additional
    `num_classes` argument, controlling the number of predicted output classes.

    To fine-tune with `fit()`, pass a dataset containing tuples of `(x, y)`
    labels where `x` is a string and `y` is a integer from `[0, num_classes)`.

    Not that unlike `keras_hub.model.ImageClassifier`, the `DeiTImageClassifier`
    we pluck out `cls_token` which is first seqence from the backbone.

    Args:
        backbone: A `keras_hub.models.DeiTBackbone` instance or a `keras.Model`.
        num_classes: int. The number of classes to predict.
        preprocessor: `None`, a `keras_hub.models.Preprocessor` instance,
            a `keras.Layer` instance, or a callable. If `None` no preprocessing
            will be applied to the inputs.
        pooling: String specifying the classification strategy. The choice
            impacts the dimensionality and nature of the feature vector used for
            classification.
                `"token"`:  A single vector (class token) representing the
                    overall image features.
                `"gap"`: A single vector representing the average features
                    across the spatial dimensions.
        activation: `None`, str, or callable. The activation function to use on
            the `Dense` layer. Set `activation=None` to return the output
            logits. Defaults to `None`.
        head_dtype: `None`, str, or `keras.mixed_precision.DTypePolicy`. The
            dtype to use for the classification head's computations and weights.

    Examples:

    Call `predict()` to run inference.
    ```python
    # Load preset and train
    images = np.random.randint(0, 256, size=(2, 384, 384, 3))
    classifier = keras_hub.models.DeiTImageClassifier.from_preset(
        "hf://facebook/deit-base-distilled-patch16-384"
    )
    classifier.predict(images)
    ```

    Call `fit()` on a single batch.
    ```python
    # Load preset and train
    images = np.random.randint(0, 256, size=(2, 384, 384, 3))
    labels = [0, 3]
    classifier = keras_hub.models.DeiTImageClassifier.from_preset(
        "hf://facebook/deit-base-distilled-patch16-384"
    )
    classifier.fit(x=images, y=labels, batch_size=2)
    ```

    Call `fit()` with custom loss, optimizer and backbone.
    ```python
    classifier = keras_hub.models.DeiTImageClassifier.from_preset(
        "hf://facebook/deit-base-distilled-patch16-384"
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
    images = np.random.randint(0, 256, size=(2, 384, 384, 3))
    labels = [0, 3]
    backbone = keras_hub.models.DeiTBackbone(
        image_shape = (384, 384, 3),
        patch_size=16,
        num_layers=6,
        num_heads=3,
        hidden_dim=768,
        intermediate_dim=2048
    )
    classifier = keras_hub.models.DeiTImageClassifier(
        backbone=backbone,
        num_classes=4,
    )
    classifier.fit(x=images, y=labels, batch_size=2)
    ```
    """

    backbone_cls = DeiTBackbone
    preprocessor_cls = DeiTImageClassifierPreprocessor

    def __init__(
        self,
        backbone,
        num_classes,
        preprocessor=None,
        pooling="token",
        activation=None,
        dropout=0.0,
        head_dtype=None,
        **kwargs,
    ):
        head_dtype = head_dtype or backbone.dtype_policy

        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor
        self.dropout = keras.layers.Dropout(
            rate=dropout,
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
        if pooling == "token":
            x = x[:, 0]
        elif pooling == "gap":
            ndim = len(ops.shape(x))
            x = ops.mean(x, axis=list(range(1, ndim - 1)))  # (1,) or (1,2)

        outputs = self.output_dense(x)

        # Skip the parent class functional model.
        Task.__init__(
            self,
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

        # === config ===
        self.num_classes = num_classes
        self.pooling = pooling
        self.activation = activation
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
