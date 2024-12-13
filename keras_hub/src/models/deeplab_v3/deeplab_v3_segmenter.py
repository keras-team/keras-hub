import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.deeplab_v3.deeplab_v3_backbone import (
    DeepLabV3Backbone,
)
from keras_hub.src.models.deeplab_v3.deeplab_v3_image_segmeter_preprocessor import (  # noqa: E501
    DeepLabV3ImageSegmenterPreprocessor,
)
from keras_hub.src.models.image_segmenter import ImageSegmenter


@keras_hub_export("keras_hub.models.DeepLabV3ImageSegmenter")
class DeepLabV3ImageSegmenter(ImageSegmenter):
    """DeepLabV3 and DeeplabV3 and DeeplabV3Plus segmentation task.

    Args:
        backbone: A `keras_hub.models.DeepLabV3` instance.
        num_classes: int. The number of classes for the detection model. Note
            that the `num_classes` contains the background class, and the
            classes from the data should be represented by integers with range
            `[0, num_classes]`.
        activation: str or callable. The activation function to use on
            the `Dense` layer. Set `activation=None` to return the output
            logits. Defaults to `None`.
        preprocessor: A `keras_hub.models.DeepLabV3ImageSegmenterPreprocessor`
            or `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model.

    Example:
    Load a DeepLabV3 preset with all the 21 class, pretrained segmentation head.
    ```python
    images = np.ones(shape=(1, 96, 96, 3))
    labels = np.zeros(shape=(1, 96, 96, 2))
    segmenter = keras_hub.models.DeepLabV3ImageSegmenter.from_preset(
        "deeplab_v3_plus_resnet50_pascalvoc",
    )
    segmenter.predict(images)
    ```

    Specify `num_classes` to load randomly initialized segmentation head.
    ```python
    segmenter = keras_hub.models.DeepLabV3ImageSegmenter.from_preset(
        "deeplab_v3_plus_resnet50_pascalvoc",
        num_classes=2,
    )
    segmenter.preprocessor.image_size = (96, 96)
    segmenter.fit(images, labels, epochs=3)
    segmenter.predict(images)  # Trained 2 class segmentation.
    ```

    Load DeepLabv3+ presets a extension of DeepLabv3 by adding a simple yet
    effective decoder module to refine the segmentation results especially
    along object boundaries.
    ```python
    segmenter = keras_hub.models.DeepLabV3ImageSegmenter.from_preset(
        "deeplabv3_plus_resnet50_pascalvoc",
    )
    segmenter.predict(images)
    ```
    """

    backbone_cls = DeepLabV3Backbone
    preprocessor_cls = DeepLabV3ImageSegmenterPreprocessor

    def __init__(
        self,
        backbone,
        num_classes,
        activation=None,
        preprocessor=None,
        **kwargs,
    ):
        data_format = keras.config.image_data_format()
        # === Layers ===
        self.output_conv = keras.layers.Conv2D(
            name="segmentation_output",
            filters=num_classes,
            kernel_size=1,
            use_bias=False,
            padding="same",
            activation=activation,
            data_format=data_format,
        )

        # === Functional Model ===
        inputs = backbone.input
        x = backbone(inputs)
        outputs = self.output_conv(x)
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

        # === Config ===
        self.backbone = backbone
        self.num_classes = num_classes
        self.activation = activation
        self.preprocessor = preprocessor

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
