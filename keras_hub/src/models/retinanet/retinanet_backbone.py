import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.feature_pyramid_backbone import FeaturePyramidBackbone
from keras_hub.src.models.retinanet.feature_pyramid import FeaturePyramid
from keras_hub.src.utils.keras_utils import standardize_data_format


@keras_hub_export("keras_hub.models.RetinaNetBackbone")
class RetinaNetBackbone(FeaturePyramidBackbone):
    def __init__(
        self,
        backbone,
        min_level,
        max_level,
        image_shape=(None, None, 3),
        data_format=None,
        dtype=None,
        **kwargs,
    ):

        if min_level > max_level:
            raise ValueError(
                f"Minimum level ({min_level}) must be less than or equal to "
                f"maximum level ({max_level})."
            )

        data_format = standardize_data_format(data_format)
        input_levels = [int(level[1]) for level in backbone.pyramid_outputs]
        backbone_max_level = min(max(input_levels), max_level)
        image_encoder = keras.Model(
            inputs=backbone.input,
            outputs={
                f"P{i}": backbone.pyramid_outputs[f"P{i}"]
                for i in range(min_level, backbone_max_level + 1)
            },
            name="backbone",
        )

        feature_pyramid = FeaturePyramid(
            min_level=min_level, max_level=max_level, name="fpn", dtype=dtype
        )

        # === Functional model ===
        image_input = keras.layers.Input(image_shape, name="images")

        image_encoder_outputs = image_encoder(image_input)
        feature_pyramid_outputs = feature_pyramid(image_encoder_outputs)

        # === config ===
        self.min_level = min_level
        self.max_level = max_level
        self.backbone = backbone
        self.feature_pyramid = feature_pyramid
        self.image_shape = image_shape

        super().__init__(
            inputs=image_input,
            outputs=feature_pyramid_outputs,
            dtype=dtype,
            **kwargs,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone": keras.layers.serialize(self.backbone),
                "min_level": self.min_level,
                "max_level": self.max_level,
                "image_shape": self.image_shape,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config.update(
            {
                "backbone": keras.layers.deserialize(config["backbone"]),
            }
        )

        return super().from_config(config)
