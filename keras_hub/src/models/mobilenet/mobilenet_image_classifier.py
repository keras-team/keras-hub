import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.mobilenet.mobilenet_backbone import MobileNetBackbone
from keras_hub.src.models.mobilenet.mobilenet_image_classifier_preprocessor \
import MobileNetImageClassifierPreprocessor


@keras_hub_export("keras_hub.models.MobileNetImageClassifier")
class MobileNetImageClassifier(ImageClassifier):
    backbone_cls = MobileNetBackbone
    preprocessor_cls = MobileNetImageClassifierPreprocessor

    def __init__(
        self,
        backbone,
        num_classes,
        preprocessor=None,
        head_dtype=None,
        **kwargs,
    ):
        super().__init__(
            backbone,
            num_classes,
            preprocessor=preprocessor,
            head_dtype=head_dtype,
            **kwargs,
        )

        head_dtype = head_dtype or backbone.dtype_policy
        data_format = getattr(backbone, "data_format", None)

        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor
        self.pooler = keras.layers.GlobalAveragePooling2D(
            data_format, keepdims=True, dtype=head_dtype, name="pooler"
        )

        self.output_conv = keras.layers.Conv2D(
            filters=1024,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=True,
            padding="valid",
            activation="hard_silu",
        )

        self.flatten = keras.layers.Flatten()

        self.output_dense = keras.layers.Dense(
            num_classes,
            dtype=head_dtype,
            name="predictions",
        )

        # === Config ===
        self.num_classes = num_classes

    def get_config(self):
        # Backbone serialized in `super`
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
            }
        )
        return config

    def call(self, inputs):
        x = self.backbone(inputs)
        x = self.pooler(x)
        x = self.output_conv(x)
        x = self.flatten(x)
        x = self.output_dense(x)
        return x
