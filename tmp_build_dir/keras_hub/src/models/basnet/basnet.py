import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.basnet.basnet_backbone import BASNetBackbone
from keras_hub.src.models.basnet.basnet_preprocessor import BASNetPreprocessor
from keras_hub.src.models.image_segmenter import ImageSegmenter


@keras_hub_export("keras_hub.models.BASNetImageSegmenter")
class BASNetImageSegmenter(ImageSegmenter):
    """BASNet image segmentation task.

    Args:
        backbone: A `keras_hub.models.BASNetBackbone` instance.
        preprocessor: `None`, a `keras_hub.models.Preprocessor` instance,
            a `keras.Layer` instance, or a callable. If `None` no preprocessing
            will be applied to the inputs.

    Example:
    ```python
    import keras_hub

    images = np.ones(shape=(1, 288, 288, 3))
    labels = np.zeros(shape=(1, 288, 288, 1))

    image_encoder = keras_hub.models.ResNetBackbone.from_preset(
        "resnet_18_imagenet",
        load_weights=False
    )
    backbone = keras_hub.models.BASNetBackbone(
        image_encoder,
        num_classes=1,
        image_shape=[288, 288, 3]
    )
    model = keras_hub.models.BASNetImageSegmenter(backbone)

    # Evaluate the model
    pred_labels = model(images)

    # Train the model
    model.compile(
        optimizer="adam",
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    model.fit(images, labels, epochs=3)
    ```
    """

    backbone_cls = BASNetBackbone
    preprocessor_cls = BASNetPreprocessor

    def __init__(
        self,
        backbone,
        preprocessor=None,
        **kwargs,
    ):
        # === Functional Model ===
        x = backbone.input
        outputs = backbone(x)
        # only return the refinement module's output as final prediction
        outputs = outputs["refine_out"]
        super().__init__(inputs=x, outputs=outputs, **kwargs)

        # === Config ===
        self.backbone = backbone
        self.preprocessor = preprocessor

    def compute_loss(self, x, y, y_pred, *args, **kwargs):
        # train BASNet's prediction and refinement module outputs against the
        # same ground truth data
        outputs = self.backbone(x)
        losses = []
        for output in outputs.values():
            losses.append(super().compute_loss(x, y, output, *args, **kwargs))
        return keras.ops.sum(losses, axis=0)

    def compile(
        self,
        optimizer="auto",
        loss="auto",
        metrics="auto",
        **kwargs,
    ):
        """Configures the `BASNet` task for training.

        `BASNet` extends the default compilation signature
        of `keras.Model.compile` with defaults for `optimizer` and `loss`. To
        override these defaults, pass any value to these arguments during
        compilation.

        Args:
            optimizer: `"auto"`, an optimizer name, or a `keras.Optimizer`
                instance. Defaults to `"auto"`, which uses the default
                optimizer for `BASNet`. See `keras.Model.compile` and
                `keras.optimizers` for more info on possible `optimizer`
                values.
            loss: `"auto"`, a loss name, or a `keras.losses.Loss` instance.
                Defaults to `"auto"`, in which case the default loss
                computation of `BASNet` will be applied.
                See `keras.Model.compile` and `keras.losses` for more info on
                possible `loss` values.
            metrics: `"auto"`, or a list of metrics to be evaluated by
                the model during training and testing. Defaults to `"auto"`,
                where a `keras.metrics.Accuracy` will be applied to track the
                accuracy of the model during training.
                See `keras.Model.compile` and `keras.metrics` for
                more info on possible `metrics` values.
            **kwargs: See `keras.Model.compile` for a full list of arguments
                supported by the compile method.
        """
        if loss == "auto":
            loss = keras.losses.BinaryCrossentropy()
        if metrics == "auto":
            metrics = [keras.metrics.Accuracy()]
        super().compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            **kwargs,
        )
