import keras
from keras import layers

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.depth_anything.depth_anything_layers import (
    DepthAnythingDepthEstimationHead,
)
from keras_hub.src.models.depth_anything.depth_anything_layers import (
    DepthAnythingNeck,
)
from keras_hub.src.models.dinov2 import DINOV2Backbone
from keras_hub.src.utils.keras_utils import standardize_data_format


@keras_hub_export("keras_hub.models.DepthAnythingBackbone")
class DepthAnythingBackbone(Backbone):
    def __init__(
        self,
        image_encoder,
        patch_size,
        backbone_hidden_dim,
        reassemble_factors,
        neck_hidden_dims,
        fusion_hidden_dim,
        head_hidden_dim,
        head_in_index,
        image_shape=(224, 224, 3),
        feature_keys=None,
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        if not isinstance(image_encoder, DINOV2Backbone):
            raise ValueError(
                "`image_encoder` must be a `DINOV2Backbone`. "
                f"Received image_encoder={image_encoder} "
                f"(of type {type(image_encoder)})."
            )
        if feature_keys is not None:
            feature_keys = [str(key) for key in feature_keys]
            for key in feature_keys:
                if key not in image_encoder.pyramid_outputs:
                    raise ValueError(
                        "All `feature_keys` must be in "
                        "`image_encoder.pyramid_outputs`. "
                        f"Received feature_keys={feature_keys}, but "
                        "`image_encoder.pyramid_outputs` contains "
                        f"{list(image_encoder.pyramid_outputs.keys())}."
                    )
        data_format = standardize_data_format(data_format)
        if data_format == "channels_last":
            image_size = (image_shape[0], image_shape[1])
        else:
            image_size = (image_shape[1], image_shape[2])

        # === Layers ===
        if feature_keys is None:
            pyramid_outputs = image_encoder.pyramid_outputs
        else:
            pyramid_outputs = {
                key: value
                for key, value in image_encoder.pyramid_outputs.items()
                if key in feature_keys
            }
        self.feature_extractor = keras.Model(
            inputs=image_encoder.inputs,
            outputs=pyramid_outputs,
        )
        self.feature_extractor.dtype_policy = image_encoder.dtype_policy
        self.neck = DepthAnythingNeck(
            patch_size=patch_size,
            image_size=image_size,
            backbone_hidden_dim=backbone_hidden_dim,
            neck_hidden_dims=neck_hidden_dims,
            reassemble_factors=reassemble_factors,
            fusion_hidden_dim=fusion_hidden_dim,
            num_cls_tokens=1,
            num_register_tokens=image_encoder.num_register_tokens,
            data_format=data_format,
            dtype=dtype,
            name="neck",
        )
        self.head = DepthAnythingDepthEstimationHead(
            patch_size=patch_size,
            patch_height=image_size[0] // patch_size,
            patch_width=image_size[1] // patch_size,
            fusion_hidden_dim=fusion_hidden_dim,
            head_hidden_dim=head_hidden_dim,
            head_in_index=head_in_index,
            data_format=data_format,
            dtype=dtype,
            name="head",
        )

        # === Functional Model ===
        image_input = layers.Input(shape=image_shape, name="images")
        features = self.feature_extractor(image_input)
        features = self.neck(list(features.values()))
        depth_output = self.head(features)
        super().__init__(
            inputs=image_input,
            outputs=depth_output,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.image_encoder = image_encoder
        self.patch_size = patch_size
        self.backbone_hidden_dim = backbone_hidden_dim
        self.reassemble_factors = reassemble_factors
        self.neck_hidden_dims = neck_hidden_dims
        self.fusion_hidden_dim = fusion_hidden_dim
        self.head_hidden_dim = head_hidden_dim
        self.head_in_index = head_in_index
        self.image_shape = image_shape
        self.feature_keys = feature_keys

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_encoder": layers.serialize(self.image_encoder),
                "patch_size": self.patch_size,
                "backbone_hidden_dim": self.backbone_hidden_dim,
                "reassemble_factors": self.reassemble_factors,
                "neck_hidden_dims": self.neck_hidden_dims,
                "fusion_hidden_dim": self.fusion_hidden_dim,
                "head_hidden_dim": self.head_hidden_dim,
                "head_in_index": self.head_in_index,
                "image_shape": self.image_shape,
                "feature_keys": self.feature_keys,
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = config.copy()

        # Propagate `dtype` to `image_encoder` if needed.
        if "dtype" in config and config["dtype"] is not None:
            dtype_config = config["dtype"]
            if "dtype" not in config["image_encoder"]["config"]:
                config["image_encoder"]["config"]["dtype"] = dtype_config

        # We expect submodels to be instantiated.
        config["image_encoder"] = layers.deserialize(
            config["image_encoder"], custom_objects=custom_objects
        )
        return cls(**config)
