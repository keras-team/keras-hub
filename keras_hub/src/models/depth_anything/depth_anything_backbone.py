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
    """DepthAnything core network with hyperparameters.

    DepthAnything offers a powerful monocular depth estimation as described in
    [Depth Anything V2](https://arxiv.org/abs/2406.09414).

    The default constructor gives a fully customizable, randomly initialized
    DepthAnything model with any number of layers, heads, and embedding
    dimensions by providing the DINOV2 as the `image_encoder`. To load preset
    architectures and weights, use the `from_preset` constructor.

    Args:
        image_encoder: The DINOV2 image encoder for encoding the input images.
        reassemble_factors: List of float. The reassemble factor for each
            feature map from the image encoder. The length of the list must be
            equal to the number of feature maps from the image encoder.
        neck_hidden_dims: int. The size of the neck hidden state.
        fusion_hidden_dim: int. The size of the fusion hidden state.
        head_hidden_dim: int. The size of the neck hidden state.
        head_in_index: int. The index to select the feature from the neck
            features as the input to the head.
        feature_keys: List of string. The keys to select the feature maps from
            the image encoder. If `None`, all feature maps from the image
            encoder will be used. Defaults to `None`.
        data_format: `None` or str. If specified, either `"channels_last"` or
            `"channels_first"`. The ordering of the dimensions in the
            inputs. `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the models computations and weights. Note that some
            computations, such as softmax and layer normalization will always
            be done a float32 precision regardless of dtype.

    Example:
    ```python
    # Pretrained DepthAnything model.
    input_data = {
        "images": np.ones(shape=(1, 518, 518, 3), dtype="float32"),
    }
    model = keras_hub.models.DepthAnythingBackbone.from_preset(
        "depth_anything_v2_small"
    )
    model(input_data)

    # Pretrained DepthAnything model with custom image shape.
    input_data = {
        "images": np.ones(shape=(1, 224, 224, 3), dtype="float32"),
    }
    model = keras_hub.models.DepthAnythingBackbone.from_preset(
        "depth_anything_v2_small", image_shape=(224, 224, 3)
    )
    model(input_data)

    # Randomly initialized DepthAnything model with custom config.
    image_encoder = keras_hub.models.DINOV2Backbone(
        patch_size=14,
        num_layers=4,
        hidden_dim=32,
        num_heads=2,
        intermediate_dim=128,
        image_shape=(224, 224, 3),
        position_embedding_shape=(518, 518),
    )
    model = keras_hub.models.DepthAnythingBackbone(
        image_encoder=image_encoder,
        reassemble_factors=[4, 2, 1, 0.5],
        neck_hidden_dims=[16, 32, 64, 128],
        fusion_hidden_dim=128,
        head_hidden_dim=16,
        head_in_index=-1,
        feature_keys=["Stage1", "Stage2", "Stage3", "Stage4"],
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        image_encoder,
        reassemble_factors,
        neck_hidden_dims,
        fusion_hidden_dim,
        head_hidden_dim,
        head_in_index,
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
        else:
            feature_keys = list(image_encoder.pyramid_outputs.keys())
        if len(reassemble_factors) != len(feature_keys):
            raise ValueError(
                "The length of `reassemble_factors` must be equal to the "
                "length of `feature_keys`. "
                f"Received len(reassemble_factors)={len(reassemble_factors)}, "
                f"len(feature_keys)={len(feature_keys)}."
            )
        data_format = standardize_data_format(data_format)
        patch_size = image_encoder.patch_size
        backbone_hidden_dim = image_encoder.hidden_dim
        image_shape = image_encoder.image_shape
        if data_format == "channels_last":
            image_size = (image_shape[0], image_shape[1])
        else:
            image_size = (image_shape[1], image_shape[2])

        # === Layers ===
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
        self.reassemble_factors = reassemble_factors
        self.neck_hidden_dims = neck_hidden_dims
        self.fusion_hidden_dim = fusion_hidden_dim
        self.head_hidden_dim = head_hidden_dim
        self.head_in_index = head_in_index
        self.feature_keys = feature_keys

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_encoder": layers.serialize(self.image_encoder),
                "reassemble_factors": self.reassemble_factors,
                "neck_hidden_dims": self.neck_hidden_dims,
                "fusion_hidden_dim": self.fusion_hidden_dim,
                "head_hidden_dim": self.head_hidden_dim,
                "head_in_index": self.head_in_index,
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
