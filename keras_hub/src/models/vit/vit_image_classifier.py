import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.task import Task
from keras_hub.src.models.vit.vit_backbone import ViTBackbone
from keras_hub.src.models.vit.vit_image_classifier_preprocessor import (
    ViTImageClassifierPreprocessor,
)


@keras_hub_export("keras_hub.models.ViTImageClassifier")
class ViTImageClassifier(ImageClassifier):
    """ViT image classification task.

    `ViTImageClassifier` tasks wrap a `keras_hub.models.ViTBackbone` and
    a `keras_hub.models.Preprocessor` to create a model that can be used for
    image classification. `ViTImageClassifier` tasks take an additional
    `num_classes` argument, controlling the number of predicted output classes.

    To fine-tune with `fit()`, pass a dataset containing tuples of `(x, y)`
    labels where `x` is a string and `y` is a integer from `[0, num_classes)`.

    Not that unlike `keras_hub.model.ImageClassifier`, the `ViTImageClassifier`
    we pluck out `cls_token` which is first seqence from the backbone.

    Args:
        backbone: A `keras_hub.models.ViTBackbone` instance or a `keras.Model`.
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
        intermediate_dim: Optional dimensionality of the intermediate
            representation layer before the final classification layer.
            If `None`, the output of the transformer is directly used.
            Defaults to `None`.
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
    classifier = keras_hub.models.ViTImageClassifier.from_preset(
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
        "vit_base_patch16_224"
    )
    classifier.fit(x=images, y=labels, batch_size=2)
    ```

    Call `fit()` with custom loss, optimizer and backbone.
    ```python
    classifier = keras_hub.models.VGGImageClassifier.from_preset(
        "vit_base_patch16_224"
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
    model = keras_hub.models.ViTBackbone(
        image_shape = (224, 224, 3),
        patch_size=16,
        num_layers=6,
        num_heads=3,
        hidden_dim=768,
        mlp_dim=2048
    )
    classifier = keras_hub.models.ViTImageClassifier(
        backbone=backbone,
        num_classes=4,
    )
    classifier.fit(x=images, y=labels, batch_size=2)
    ```
    """

    backbone_cls = ViTBackbone
    preprocessor_cls = ViTImageClassifierPreprocessor

    def __init__(
        self,
        backbone,
        num_classes,
        preprocessor=None,
        pooling="token",
        intermediate_dim=None,
        activation=None,
        dropout=0.0,
        head_dtype=None,
        **kwargs,
    ):
        head_dtype = head_dtype or backbone.dtype_policy

        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor

        if intermediate_dim is not None:
            self.intermediate_layer = keras.layers.Dense(
                intermediate_dim, activation="tanh", name="pre_logits"
            )

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

        if intermediate_dim is not None:
            x = self.intermediate_layer(x)

        x = self.dropout(x)
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
        self.intermediate_dim = intermediate_dim
        self.activation = activation
        self.dropout = dropout

    def get_config(self):
        # Backbone serialized in `super`
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "pooling": self.pooling,
                "intermediate_dim": self.intermediate_dim,
                "activation": self.activation,
                "dropout": self.dropout,
            }
        )
        return config
