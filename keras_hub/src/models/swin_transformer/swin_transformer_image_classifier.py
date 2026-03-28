import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.swin_transformer.swin_transformer_backbone import (
    SwinTransformerBackbone,
)
from keras_hub.src.models.swin_transformer.swin_transformer_image_classifier_preprocessor import (  # noqa: E501
    SwinTransformerImageClassifierPreprocessor,
)
from keras_hub.src.models.task import Task


@keras_hub_export("keras_hub.models.SwinTransformerImageClassifier")
class SwinTransformerImageClassifier(ImageClassifier):
    """Swin Transformer image classification task.

    `SwinTransformerImageClassifier` tasks wrap a
    `keras_hub.models.SwinTransformerBackbone` and a
    `keras_hub.models.Preprocessor` to create a model that can be used
    for image classification. The classifier pools the backbone output
    over the sequence dimension and applies a dense classification head.

    To fine-tune with `fit()`, pass a dataset containing tuples of
    `(x, y)` labels where `x` is an image and `y` is an integer from
    `[0, num_classes)`.

    Args:
        backbone: A `keras_hub.models.SwinTransformerBackbone` instance
            or a `keras.Model`.
        num_classes: int. The number of classes to predict.
        preprocessor: `None`, a `keras_hub.models.Preprocessor`
            instance, a `keras.Layer` instance, or a callable. If
            `None` no preprocessing will be applied to the inputs.
        activation: `None`, str, or callable. The activation function
            to use on the `Dense` layer. Set `activation=None` to
            return the output logits. Defaults to `None`.
        head_dtype: `None`, str, or
            `keras.mixed_precision.DTypePolicy`. The dtype to use for
            the classification head's computations and weights.
        dropout: float. Dropout rate applied before the classification
            head. Defaults to `0.0`.

    Examples:

    Call `predict()` to run inference.
    ```python
    images = np.random.randint(0, 256, size=(2, 224, 224, 3))
    classifier = keras_hub.models.SwinTransformerImageClassifier.from_preset(
        "swin_tiny_patch4_window7_224"
    )
    classifier.predict(images)
    ```

    Call `fit()` on a single batch.
    ```python
    images = np.random.randint(0, 256, size=(2, 224, 224, 3))
    labels = [0, 3]
    classifier = keras_hub.models.SwinTransformerImageClassifier.from_preset(
        "swin_tiny_patch4_window7_224"
    )
    classifier.fit(x=images, y=labels, batch_size=2)
    ```

    Custom backbone.
    ```python
    images = np.random.randint(0, 256, size=(2, 224, 224, 3))
    labels = [0, 3]
    backbone = keras_hub.models.SwinTransformerBackbone(
        image_shape=(224, 224, 3),
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
    )
    classifier = keras_hub.models.SwinTransformerImageClassifier(
        backbone=backbone,
        num_classes=4,
    )
    classifier.fit(x=images, y=labels, batch_size=2)
    ```
    """

    backbone_cls = SwinTransformerBackbone
    preprocessor_cls = SwinTransformerImageClassifierPreprocessor

    def __init__(
        self,
        backbone,
        num_classes,
        preprocessor=None,
        activation=None,
        dropout=0.0,
        head_dtype=None,
        **kwargs,
    ):
        head_dtype = head_dtype or backbone.dtype_policy

        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor
        self.output_dropout = keras.layers.Dropout(
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
        x = ops.mean(x, axis=1)
        x = self.output_dropout(x)
        outputs = self.output_dense(x)

        Task.__init__(
            self,
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

        # === Config ===
        self.num_classes = num_classes
        self.pooling = "gap"
        self.activation = activation
        self.dropout = dropout
