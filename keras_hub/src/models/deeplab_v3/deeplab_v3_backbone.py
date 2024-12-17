import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.deeplab_v3.deeplab_v3_layers import (
    SpatialPyramidPooling,
)


@keras_hub_export("keras_hub.models.DeepLabV3Backbone")
class DeepLabV3Backbone(Backbone):
    """DeepLabV3 & DeepLabV3Plus architecture for semantic segmentation.

    This class implements a DeepLabV3 & DeepLabV3Plus architecture as described
    in [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)
    (ECCV 2018)
    and [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)
    (CVPR 2017)

    Args:
        image_encoder: `keras.Model`. An instance that is used as a feature
            extractor for the Encoder. Should either be a
            `keras_hub.models.Backbone` or a `keras.Model` that implements the
            `pyramid_outputs` property with keys "P2", "P3" etc as values.
            A somewhat sensible backbone to use in many cases is
            the `keras_hub.models.ResNetBackbone.from_preset("resnet_v2_50")`.
        projection_filters: int. Number of filters in the convolution layer
            projecting low-level features from the `image_encoder`.
        spatial_pyramid_pooling_key: str. A layer level to extract and perform
            `spatial_pyramid_pooling`, one of the key from the `image_encoder`
            `pyramid_outputs` property such as  "P4", "P5" etc.
        upsampling_size: int or tuple of 2 integers. The upsampling factors for
            rows and columns of `spatial_pyramid_pooling` layer.
            If `low_level_feature_key` is given then `spatial_pyramid_pooling`s
            layer resolution should match with the `low_level_feature`s layer
            resolution to concatenate both the layers for combined encoder
            outputs.
        dilation_rates: list. A `list` of integers for parallel dilated conv
            applied to `SpatialPyramidPooling`. Usually a
            sample choice of rates are `[6, 12, 18]`.
        low_level_feature_key: str optional. A layer level to extract the
            feature from one of the key from the `image_encoder`s
            `pyramid_outputs` property such as  "P2", "P3" etc which will be the
            Decoder block. Required only when the DeepLabV3Plus architecture
            needs to be applied.
        image_shape: tuple. The input shape without the batch size.
            Defaults to `(None, None, 3)`.

    Example:
    ```python
    # Load a trained backbone to extract features from it's `pyramid_outputs`.
    image_encoder = keras_hub.models.ResNetBackbone.from_preset(
        "resnet_50_imagenet"
    )

    model = keras_hub.models.DeepLabV3Backbone(
        image_encoder=image_encoder,
        projection_filters=48,
        low_level_feature_key="P2",
        spatial_pyramid_pooling_key="P5",
        upsampling_size = 8,
        dilation_rates = [6, 12, 18]
    )
    ```
    """  # noqa: E501

    def __init__(
        self,
        image_encoder,
        spatial_pyramid_pooling_key,
        upsampling_size,
        dilation_rates,
        low_level_feature_key=None,
        projection_filters=48,
        image_shape=(None, None, 3),
        **kwargs,
    ):
        if not isinstance(image_encoder, keras.Model):
            raise ValueError(
                "Argument `image_encoder` must be a `keras.Model` instance. "
                "Received instead "
                f"{image_encoder} (of type {type(image_encoder)})."
            )
        data_format = keras.config.image_data_format()
        channel_axis = -1 if data_format == "channels_last" else 1

        # === Layers ===
        inputs = keras.layers.Input(image_shape, name="inputs")

        fpn_model = keras.Model(
            image_encoder.inputs, image_encoder.pyramid_outputs
        )

        fpn_outputs = fpn_model(inputs)

        spatial_pyramid_pooling = SpatialPyramidPooling(
            dilation_rates=dilation_rates
        )
        spatial_backbone_features = fpn_outputs[spatial_pyramid_pooling_key]
        spp_outputs = spatial_pyramid_pooling(spatial_backbone_features)

        encoder_outputs = keras.layers.UpSampling2D(
            size=upsampling_size,
            interpolation="bilinear",
            name="encoder_output_upsampling",
            data_format=data_format,
        )(spp_outputs)

        if low_level_feature_key:
            decoder_feature = fpn_outputs[low_level_feature_key]
            low_level_projected_features = apply_low_level_feature_network(
                decoder_feature, projection_filters, channel_axis
            )

            encoder_outputs = keras.layers.Concatenate(
                axis=channel_axis, name="encoder_decoder_concat"
            )([encoder_outputs, low_level_projected_features])
        # upsampling to the original image size
        upsampling = (2 ** int(spatial_pyramid_pooling_key[-1])) // (
            int(upsampling_size[0])
            if isinstance(upsampling_size, tuple)
            else upsampling_size
        )
        # === Functional Model ===
        x = keras.layers.Conv2D(
            name="segmentation_head_conv",
            filters=256,
            kernel_size=1,
            padding="same",
            use_bias=False,
            data_format=data_format,
        )(encoder_outputs)
        x = keras.layers.BatchNormalization(
            name="segmentation_head_norm", axis=channel_axis
        )(x)
        x = keras.layers.ReLU(name="segmentation_head_relu")(x)
        x = keras.layers.UpSampling2D(
            size=upsampling,
            interpolation="bilinear",
            data_format=data_format,
            name="backbone_output_upsampling",
        )(x)

        super().__init__(inputs=inputs, outputs=x, **kwargs)

        # === Config ===
        self.image_shape = image_shape
        self.image_encoder = image_encoder
        self.projection_filters = projection_filters
        self.upsampling_size = upsampling_size
        self.dilation_rates = dilation_rates
        self.low_level_feature_key = low_level_feature_key
        self.spatial_pyramid_pooling_key = spatial_pyramid_pooling_key

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_encoder": keras.saving.serialize_keras_object(
                    self.image_encoder
                ),
                "projection_filters": self.projection_filters,
                "dilation_rates": self.dilation_rates,
                "upsampling_size": self.upsampling_size,
                "low_level_feature_key": self.low_level_feature_key,
                "spatial_pyramid_pooling_key": self.spatial_pyramid_pooling_key,
                "image_shape": self.image_shape,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if "image_encoder" in config and isinstance(
            config["image_encoder"], dict
        ):
            config["image_encoder"] = keras.layers.deserialize(
                config["image_encoder"]
            )
        return super().from_config(config)


def apply_low_level_feature_network(
    input_tensor, projection_filters, channel_axis
):
    data_format = keras.config.image_data_format()
    x = keras.layers.Conv2D(
        name="decoder_conv",
        filters=projection_filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
    )(input_tensor)

    x = keras.layers.BatchNormalization(name="decoder_norm", axis=channel_axis)(
        x
    )
    x = keras.layers.ReLU(name="decoder_relu")(x)
    return x
