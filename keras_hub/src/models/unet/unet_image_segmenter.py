import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_segmenter import ImageSegmenter
from keras_hub.src.models.image_segmenter_preprocessor import (
    ImageSegmenterPreprocessor,
)
from keras_hub.src.models.unet.unet_backbone import UNetBackbone


@keras_hub_export("keras_hub.models.UNetImageSegmenterPreprocessor")
class UNetImageSegmenterPreprocessor(ImageSegmenterPreprocessor):
    """Preprocessor for UNet image segmentation.

    This preprocessor simply passes through the input images and labels
    without any modification, since UNet can handle variable input sizes.
    """

    def __init__(self, **kwargs):
        super().__init__(image_converter=None, **kwargs)

    def call(self, x, y=None, sample_weight=None):
        # For UNet, we don't need any preprocessing - just pass through
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)


@keras_hub_export("keras_hub.models.UNetImageSegmenter")
class UNetImageSegmenter(ImageSegmenter):
    """UNet image segmentation task.

    Args:
        backbone: A `keras_hub.models.UNetBackbone` instance.
        num_classes: int. The number of classes for the segmentation model.
            Note that the `num_classes` contains the background class, and the
            classes from the data should be represented by integers with range
            `[0, num_classes)`.
        activation: str or callable. The activation function to use on the
            output layer. Set `activation=None` to return the output logits.
            Defaults to `"softmax"`.
        preprocessor: A preprocessor instance or `None`. If `None`, this model
            will not apply preprocessing, and inputs should be preprocessed
            before calling the model.

    Example:
    ```python
    import numpy as np
    import keras_hub

    # Create a UNet segmenter with a randomly initialized backbone
    backbone = keras_hub.models.UNetBackbone(
        depth=4,
        filters=64,
        image_shape=(None, None, 3),
    )
    segmenter = keras_hub.models.UNetImageSegmenter(
        backbone=backbone,
        num_classes=2,
    )

    # The model accepts any image size
    images = np.random.uniform(0, 1, size=(2, 256, 256, 3))
    labels = np.random.randint(0, 2, size=(2, 256, 256, 2))

    segmenter.compile(optimizer='adam', loss='categorical_crossentropy')
    segmenter.fit(images, labels, epochs=1)

    # Can predict on different sizes
    images = np.random.uniform(0, 1, size=(1, 512, 512, 3))
    predictions = segmenter.predict(images)
    ```
    """

    backbone_cls = UNetBackbone

    def __init__(
        self,
        backbone,
        num_classes,
        activation="softmax",
        preprocessor=None,
        **kwargs,
    ):
        if not isinstance(backbone, UNetBackbone):
            raise ValueError(
                f"backbone must be a UNetBackbone instance. "
                f"Received: backbone={backbone} (of type {type(backbone)})."
            )

        data_format = backbone.data_format

        # === Layers ===
        self.output_conv = keras.layers.Conv2D(
            name="segmentation_output",
            filters=num_classes,
            kernel_size=(1, 1),
            padding="same",
            activation=activation,
            data_format=data_format,
        )

        # === Functional Model ===
        inputs = backbone.input
        x = backbone(inputs)
        outputs = self.output_conv(x)

        if preprocessor is None:
            preprocessor = UNetImageSegmenterPreprocessor()

        # Set attributes
        self.backbone = backbone
        self.num_classes = num_classes
        self.activation = activation
        self.preprocessor = preprocessor

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            preprocessor=preprocessor,
            **kwargs,
        )

    def get_config(self):
        # Backbone serialized in `super`
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "activation": self.activation,
            }
        )
        return config
