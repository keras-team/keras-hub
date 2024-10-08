import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.feature_pyramid_backbone import FeaturePyramidBackbone
from keras_hub.src.models.retinanet.feature_pyramid import FeaturePyramid
from keras_hub.src.utils.keras_utils import standardize_data_format


@keras_hub_export("keras_hub.models.RetinaNetBackbone")
class RetinaNetBackbone(FeaturePyramidBackbone):
    """RetinaNet Backbone.

    Combines a CNN backbone (e.g., ResNet, MobileNet) with a feature pyramid
    network (FPN)to extract multi-scale features for object detection.

    Args:
        image_encoder (keras.Model): The backbone model used to extract features
            from the input image.
            It should have pyramid outputs.
        min_level (int): The minimum feature pyramid level.
        max_level (int): The maximum feature pyramid level.
        image_shape (tuple): The shape of the input image.
        data_format (str): The data format of the input image (channels_first or channels_last).
        dtype (str): The data type of the input image.
        **kwargs: Additional arguments passed to the base class.

    Raises:
        ValueError: If `min_level` is greater than `max_level`.
        ValueError: If `backbone_max_level` is less than 5 and `max_level` is greater than or equal to 5.
    """

    def __init__(
        self,
        image_encoder,
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
        input_levels = [
            int(level[1]) for level in image_encoder.pyramid_outputs
        ]
        backbone_max_level = min(max(input_levels), max_level)

        if backbone_max_level < 5 and max_level >= 5:
            raise ValueError(
                f"Backbone maximum level ({backbone_max_level}) is less than "
                f"the desired maximum level ({max_level}). "
                f"Please ensure that the backbone can generate features up to "
                f"the specified maximum level."
            )
        feature_extractor = keras.Model(
            inputs=image_encoder.inputs,
            outputs={
                f"P{level}": image_encoder.pyramid_outputs[f"P{level}"]
                for level in input_levels
            },
            name="backbone",
        )

        feature_pyramid = FeaturePyramid(
            min_level=min_level,
            max_level=max_level,
            name="fpn",
            dtype=dtype,
            data_format=data_format,
        )

        # === Functional model ===
        image_input = keras.layers.Input(image_shape, name="inputs")
        feature_extractor_outputs = feature_extractor(image_input)
        feature_pyramid_outputs = feature_pyramid(feature_extractor_outputs)

        super().__init__(
            inputs=image_input,
            outputs=feature_pyramid_outputs,
            dtype=dtype,
            **kwargs,
        )

        # === config ===
        self.min_level = min_level
        self.max_level = max_level
        self.image_encoder = image_encoder
        self.feature_pyramid = feature_pyramid
        self.image_shape = image_shape
        self.pyramid_outputs = feature_pyramid_outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_encoder": keras.layers.serialize(self.image_encoder),
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
                "image_encoder": keras.layers.deserialize(
                    config["image_encoder"]
                ),
            }
        )

        return super().from_config(config)
