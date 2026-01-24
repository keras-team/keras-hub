import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.hgnetv2.hgnetv2_backbone import HGNetV2Backbone
from keras_hub.src.models.hgnetv2.hgnetv2_image_classifier_preprocessor import (
    HGNetV2ImageClassifierPreprocessor,
)
from keras_hub.src.models.hgnetv2.hgnetv2_layers import HGNetV2ConvLayer
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.task import Task


@keras_hub_export("keras_hub.models.HGNetV2ImageClassifier")
class HGNetV2ImageClassifier(ImageClassifier):
    """HGNetV2 image classification model.

    `HGNetV2ImageClassifier` wraps a `HGNetV2Backbone` and
    a `HGNetV2ImageClassifierPreprocessor` to create a model that can be used
    for image classification tasks. This model implements the HGNetV2
    architecture with an additional classification head including a 1x1
    convolution layer, global pooling, and a dense output layer.

    The model takes an additional `num_classes` argument, controlling the number
    of predicted output classes, and optionally, a `head_filters` argument to
    specify the number of filters in the classification head's convolution
    layer. To fine-tune with `fit()`, pass a dataset containing tuples of
    `(x, y)` labels where `x` is an image tensor and `y` is an integer from
    `[0, num_classes)`.

    Args:
        backbone: A `HGNetV2Backbone` instance.
        preprocessor: A `HGNetV2ImageClassifierPreprocessor` instance,
            a `keras.Layer` instance, or a callable. If `None` no preprocessing
            will be applied to the inputs.
        num_classes: int. The number of classes to predict.
        head_filters: int, optional. The number of filters in the
            classification head's 1x1 convolution layer. If `None`, it defaults
            to the last value of `hidden_sizes` from the backbone.
        pooling: `"avg"` or `"max"`. The type of global pooling to apply after
            the head convolution. Defaults to `"avg"`.
        activation: `None`, str, or callable. The activation function to use on
            the final `Dense` layer. Set `activation=None` to return the output
            logits. Defaults to `None`.
        dropout: float. Dropout rate applied before the final dense layer.
            Defaults to 0.0.
        head_dtype: `None`, str, or `keras.mixed_precision.DTypePolicy`. The
            dtype to use for the classification head's computations and weights.

    Examples:

    Call `predict()` to run inference.
    ```python
    # Load preset and predict.
    images = np.random.randint(0, 256, size=(2, 224, 224, 3))
    classifier = keras_hub.models.HGNetV2ImageClassifier.from_preset(
        "hgnetv2_b5_ssld_stage2_ft_in1k"
    )
    classifier.predict(images)
    ```

    Call `fit()` on a single batch.
    ```python
    # Load preset and train.
    images = np.random.randint(0, 256, size=(2, 224, 224, 3))
    labels = [0, 3]
    classifier = keras_hub.models.HGNetV2ImageClassifier.from_preset(
        "hgnetv2_b5_ssld_stage2_ft_in1k"
    )
    classifier.fit(x=images, y=labels, batch_size=2)
    ```

    Call `fit()` with custom loss, optimizer and frozen backbone.
    ```python
    classifier = keras_hub.models.HGNetV2ImageClassifier.from_preset(
        "hgnetv2_b5_ssld_stage2_ft_in1k"
    )
    classifier.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(5e-5),
    )
    classifier.backbone.trainable = False
    classifier.fit(x=images, y=labels, batch_size=2)
    ```

    Create a custom HGNetV2 classifier with specific head configuration.
    ```python
    backbone = keras_hub.models.HGNetV2Backbone.from_preset(
        "hgnetv2_b5_ssld_stage2_ft_in1k"
    )
    preproc = keras_hub.models.HGNetV2ImageClassifierPreprocessor.from_preset(
        "hgnetv2_b5_ssld_stage2_ft_in1k"
    )
    classifier = keras_hub.models.HGNetV2ImageClassifier(
        backbone=backbone,
        preprocessor=preproc,
        num_classes=10,
        pooling="avg",
        dropout=0.2,
    )
    ```
    """

    backbone_cls = HGNetV2Backbone
    preprocessor_cls = HGNetV2ImageClassifierPreprocessor

    def __init__(
        self,
        backbone,
        preprocessor,
        num_classes,
        head_filters=None,
        pooling="avg",
        activation=None,
        dropout=0.0,
        head_dtype=None,
        **kwargs,
    ):
        name = kwargs.get("name", "hgnetv2_image_classifier")
        head_dtype = head_dtype or backbone.dtype_policy
        data_format = getattr(backbone, "data_format", "channels_last")
        channel_axis = -1 if data_format == "channels_last" else 1
        self.head_filters = (
            head_filters
            if head_filters is not None
            else backbone.hidden_sizes[-1]
        )
        self.activation = activation

        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor
        self.last_conv = HGNetV2ConvLayer(
            in_channels=backbone.hidden_sizes[-1],
            out_channels=self.head_filters,
            kernel_size=1,
            stride=1,
            groups=1,
            activation="relu",
            use_learnable_affine_block=self.backbone.use_learnable_affine_block,
            data_format=data_format,
            channel_axis=channel_axis,
            name="head_last",
            dtype=head_dtype,
        )
        if pooling == "avg":
            self.pooler = keras.layers.GlobalAveragePooling2D(
                data_format=data_format,
                dtype=head_dtype,
                name=f"{name}_avg_pool" if name else "avg_pool",
            )
        elif pooling == "max":
            self.pooler = keras.layers.GlobalMaxPooling2D(
                data_format=data_format,
                dtype=head_dtype,
                name=f"{name}_max_pool" if name else "max_pool",
            )
        # Check valid pooling.
        else:
            raise ValueError(
                "Unknown `pooling` type. Polling should be either `'avg'` or "
                f"`'max'`. Received: pooling={pooling}."
            )

        self.flatten_layer = keras.layers.Flatten(
            dtype=head_dtype,
            name=f"{name}_flatten" if name else "flatten",
        )
        self.output_dropout = keras.layers.Dropout(
            rate=dropout,
            dtype=head_dtype,
            name=f"{name}_output_dropout" if name else "output_dropout",
        )
        if num_classes > 0:
            self.output_dense = keras.layers.Dense(
                units=num_classes,
                activation=activation,
                dtype=head_dtype,
                name="predictions",
            )
        else:
            self.output_dense = keras.layers.Identity(name="predictions")

        # === Functional Model ===
        inputs = backbone.input
        feature_maps = backbone(inputs)
        last_stage_name = backbone.stage_names[-1]
        last_hidden_state_for_pooling = feature_maps[last_stage_name]
        x = self.last_conv(last_hidden_state_for_pooling)
        x = self.pooler(x)
        x = self.flatten_layer(x)
        x = self.output_dropout(x)
        outputs = self.output_dense(x)
        Task.__init__(
            self,
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

        # === Config ===
        self.pooling = pooling
        self.dropout = dropout
        self.num_classes = num_classes

    def get_config(self):
        config = Task.get_config(self)
        config.update(
            {
                "num_classes": self.num_classes,
                "pooling": self.pooling,
                "activation": self.activation,
                "dropout": self.dropout,
                "head_filters": self.head_filters,
            }
        )
        return config
