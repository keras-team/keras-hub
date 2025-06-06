import keras
from keras import layers

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.diffbin.db_utils import step_function
from keras_hub.src.models.diffbin.diffbin_backbone import DiffBinBackbone
from keras_hub.src.models.diffbin.diffbin_loss import DiffBinLoss
from keras_hub.src.models.diffbin.diffbin_utils import step_function
from keras_hub.src.models.image_text_detector_preprocessor import (
    ImageTextDetectorPreprocessor,
)


@keras_hub_export("keras_hub.models.DiffBinImageTextDetector")
class DiffBinTextDetector(keras.Model):
    """Differentiable Binarization scene text detection task.
    The `DiffBinImageTextDetector` class extends the `keras.Model` class and
    uses a `DiffBinBackbone` to extract features from images. It outputs
    probability maps, threshold maps, and binary maps for text detection.
    Args:
        backbone: A `DiffBinBackbone` instance that extracts features from
        images.
        preprocessor: An optional `ImageTextDetectorPreprocessor` instance for
            preprocessing input images and labels.
        **kwargs: Additional keyword arguments for the `keras.Model`.
    Examples:
        from keras_hub.src.models.diffbin.diffbin_textdetector import (
            DiffBinTextDetector,
        )
        from keras_hub.src.models.diffbin.diffbin_backbone import (
        DiffBinBackbone
        )

        backbone = DiffBinBackbone(image_encoder=my_image_encoder)
        model = DiffBinTextDetector(backbone=backbone)
    Returns:
        A `keras.Model` instance that can be used for training and inference
        on scene text detection tasks.
    """

    backbone_cls = DiffBinBackbone
    preprocessor_cls = ImageTextDetectorPreprocessor

    def __init__(
        self,
        backbone,
        preprocessor=None,
        **kwargs,
    ):
        inputs = backbone.input
        x = backbone(inputs)
        probability_maps = x["probability_maps"]
        threshold_maps = x["threshold_maps"]
        binary_maps = step_function(probability_maps, threshold_maps)
        outputs = layers.Concatenate(axis=-1)(
            [probability_maps, threshold_maps, binary_maps]
        )

        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        self.backbone = backbone
        self.preprocessor = preprocessor

    def compile(
        self,
        optimizer="auto",
        loss="auto",
        **kwargs,
    ):
        """Configures the `DiffBinImageTextDetector` task for training.

        `DiffBinImageTextDetector` extends the default compilation signature
        of `keras.Model.compile` with defaults for `optimizer` and `loss`. To
        override these defaults, pass any value to these arguments during
        compilation.

        Args:
            optimizer: `"auto"`, an optimizer name, or a `keras.Optimizer`
                instance. Defaults to `"auto"`, which uses the default
                optimizer for `DiffBinImageTextDetector`. See
                `keras.Model.compile` and `keras.optimizers` for more info on
                possible `optimizer` values.
            loss: `"auto"`, a loss name, or a `keras.losses.Loss` instance.
                Defaults to `"auto"`, in which case the default loss
                computation of `DiffBinImageTextDetector` will be applied.
                See `keras.Model.compile` and `keras.losses` for more info on
                possible `loss` values.
            **kwargs: See `keras.Model.compile` for a full list of arguments
                supported by the compile method.
        """
        if optimizer == "auto":
            # parameters from https://arxiv.org/abs/1911.08947
            optimizer = keras.optimizers.Adam(learning_rate=1e-4)
        if loss == "auto":
            loss = DiffBinLoss()
        super().compile(
            optimizer=optimizer,
            loss=loss,
            **kwargs,
        )
