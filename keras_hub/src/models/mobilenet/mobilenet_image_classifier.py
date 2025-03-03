import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.mobilenet.mobilenet_backbone import MobileNetBackbone
from keras_hub.src.models.mobilenet.mobilenet_image_classifier_preprocessor import (  # noqa: E501
    MobileNetImageClassifierPreprocessor,
)
from keras_hub.src.models.task import Task


@keras_hub_export("keras_hub.models.MobileNetImageClassifier")
class MobileNetImageClassifier(ImageClassifier):
    backbone_cls = MobileNetBackbone
    preprocessor_cls = MobileNetImageClassifierPreprocessor

    def __init__(
        self,
        backbone,
        num_classes,
        num_features=1024,
        preprocessor=None,
        head_dtype=None,
        **kwargs,
    ):
        head_dtype = head_dtype or backbone.dtype_policy
        data_format = getattr(backbone, "data_format", None)

        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor
        self.pooler = keras.layers.GlobalAveragePooling2D(
            data_format, keepdims=True, dtype=head_dtype, name="pooler"
        )

        self.output_conv = keras.layers.Conv2D(
            filters=num_features,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=True,
            padding="valid",
            activation="hard_silu",
            name="classifier_conv",
            dtype=head_dtype,
        )

        self.flatten = keras.layers.Flatten(
            dtype=head_dtype,
        )

        self.output_dense = keras.layers.Dense(
            num_classes,
            dtype=head_dtype,
            name="predictions",
        )

        # === Functional Model ===
        inputs = self.backbone.input
        x = self.backbone(inputs)
        x = self.pooler(x)
        x = self.output_conv(x)
        x = self.flatten(x)
        outputs = self.output_dense(x)
        Task.__init__(
            self,
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

        # === Config ===
        self.num_classes = num_classes
        self.num_features = num_features

    def get_config(self):
        # Skip ImageClassifier
        config = Task.get_config(self)
        config.update(
            {
                "num_classes": self.num_classes,
                "num_features": self.num_features,
            }
        )
        return config
