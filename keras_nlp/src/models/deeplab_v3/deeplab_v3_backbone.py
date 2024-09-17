# Copyright 2024 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import keras

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.models.backbone import Backbone
from keras_nlp.src.models.deeplab_v3.deeplab_v3_layers import (
    SpatialPyramidPooling,
)


@keras_nlp_export("keras_nlp.models.DeepLabV3Backbone")
class DeepLabV3Backbone(Backbone):
    """DeepLabV3 & DeepLabV3Plus architecture for semantic segmentation.

    This class implements a DeepLabV3 & DeepLabV3Plus architecture as described in
    [Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation](https://arxiv.org/abs/1802.02611)(ECCV 2018)
    and [Rethinking Atrous Convolution for Semantic Image Segmentation](
        https://arxiv.org/abs/1706.05587)(CVPR 2017)

    Args:
    image_encoder: `keras.Model`. The backbone network for the model that is
        used as a feature extractor for the Encoder. Should
        either be a `keras_nlp.models.backbones.backbone.Backbone` or a
        `keras.Model` that implements the `pyramid_outputs`
        property with keys "P2", "P3" etc as values. A
        somewhat sensible backbone to use in many cases is the
        `keras_nlp.models.ResNetBackbone.from_preset("resnet_v2_50")`.
    projection_filters: int, number of filters in the convolution layer
        projecting low-level features from the `backbone`.
    spatial_pyramid_pooling_key: str, layer level to extract and perform
        `spatial_pyramid_pooling`, one of the key from the `backbone`
        `pyramid_outputs`
        property such as  "P4", "P5" etc.
    upsampling_size: Int, or tuple of 2 integers. The upsampling factors for
        rows and columns of `spatial_pyramid_pooling` layer.
        If `low_level_feature_key` is given then `spatial_pyramid_pooling`s
        layer resolution should match with the `low_level_feature`s layer
        resolution to concatenate both the layers for combined encoder outputs.
    dilation_rates: A `list` of integers for parallel dilated conv.
        Applied only when Default `SpatialPyramidPooling` is used. Usually a
        sample choice of rates are [6, 12, 18].
    low_level_feature_key: (Optional) str, layer level to extract the feature
        from one of the key from the `backbone`s `pyramid_outputs`
        property such as  "P2", "P3" etc which will be the Decoder block.
        Required only when the DeepLabV3Plus architecture needs to be applied.
    activation: str or callable. The activation function to use on
        the `Dense` layer. Set `activation=None` to return the output
        logits. Defaults to `"softmax"`.
    spatial_pyramid_pooling: (Optional) a `keras.layers.Layer`. Also known
        as Atrous Spatial Pyramid Pooling (ASPP). Performs spatial pooling
        on different spatial levels in the pyramid, with dilation. If
        provided, the feature map from the backbone is passed to it inside
        the DeepLabV3 Encoder, otherwise SpatialPyramidPooling layer is used.
    segmentation_head: (Optional) a `keras.layers.Layer`. If provided, the
        outputs of the DeepLabV3 encoder is passed to this layer and it
        will be considered as the last layer before final segmentaion layer ,
        otherwise a default DeepLabV3 convolutional head is used.

    Example:
    ```python
    image_encoder = keras_nlp.models.ResNetBackbone.from_preset("resnet_v2_50")

    model = keras_nlp.models.DeepLabV3Backbone(
        image_encoder= image_encoder,
        projection_filters=48,
        low_level_feature_key="P2",
        spatial_pyramid_pooling_key="P5",
    )
    ```
    """

    def __init__(
        self,
        image_encoder,
        spatial_pyramid_pooling_key,
        upsampling_size,
        dilation_rates,
        low_level_feature_key=None,
        projection_filters=48,
        spatial_pyramid_pooling=None,
        segmentation_head=None,
        **kwargs,
    ):
        if not isinstance(image_encoder, keras.Model):
            raise ValueError(
                "Argument `image_encoder` must be a `keras.Model` instance. Received instead "
                f"backbone={image_encoder} (of type {type(image_encoder)})."
            )
        data_format = keras.config.image_data_format()
        channel_axis = -1 if data_format == "channels_last" else 1
        # === Functional Model ===
        inputs = keras.layers.Input((None, None, 3))

        fpn_model = keras.Model(
            image_encoder.inputs, image_encoder.pyramid_outputs
        )

        fpn_outputs = fpn_model(inputs)

        if spatial_pyramid_pooling is None:
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

            encoder_outputs = keras.layers.Concatenate(axis=channel_axis)(
                [encoder_outputs, low_level_projected_features]
            )
        # upsampling to the original image size
        upsampling = (2 ** int(spatial_pyramid_pooling_key[-1])) // (
            int(upsampling_size[0])
            if isinstance(upsampling_size, tuple)
            else upsampling_size
        )
        if segmentation_head is None:
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
            )(x)
        else:
            x = segmentation_head(encoder_outputs)

        super().__init__(inputs=inputs, outputs=x, **kwargs)

        # === Config ===
        self.image_encoder = image_encoder
        self.spatial_pyramid_pooling = spatial_pyramid_pooling
        self.projection_filters = projection_filters
        self.upsampling_size = upsampling_size
        self.segmentation_head = segmentation_head
        self.dilation_rates = dilation_rates
        self.low_level_feature_key = low_level_feature_key
        self.spatial_pyramid_pooling_key = spatial_pyramid_pooling_key

    def get_config(self):
        return {
            "image_encoder": keras.saving.serialize_keras_object(
                self.image_encoder
            ),
            "spatial_pyramid_pooling": keras.saving.serialize_keras_object(
                self.spatial_pyramid_pooling
            ),
            "projection_filters": self.projection_filters,
            "segmentation_head": keras.saving.serialize_keras_object(
                self.segmentation_head
            ),
            "dilation_rates": self.dilation_rates,
            "upsampling_size": self.upsampling_size,
            "low_level_feature_key": self.low_level_feature_key,
            "spatial_pyramid_pooling_key": self.spatial_pyramid_pooling_key,
        }

    @classmethod
    def from_config(cls, config):
        if "image_encoder" in config and isinstance(
            config["image_encoder"], dict
        ):
            config["image_encoder"] = keras.layers.deserialize(
                config["image_encoder"]
            )
        if "spatial_pyramid_pooling" in config and isinstance(
            config["spatial_pyramid_pooling"], dict
        ):
            config["spatial_pyramid_pooling"] = keras.layers.deserialize(
                config["spatial_pyramid_pooling"]
            )
        if "segmentation_head" in config and isinstance(
            config["segmentation_head"], dict
        ):
            config["segmentation_head"] = keras.layers.deserialize(
                config["segmentation_head"]
            )
        return super().from_config(config)


def apply_low_level_feature_network(
    input_tensor, projection_filters, channel_axis
):
    data_format = keras.config.image_data_format()
    x = keras.layers.Conv2D(
        name="low_level_feature_conv",
        filters=projection_filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
    )(input_tensor)

    x = keras.layers.BatchNormalization(
        name="low_level_feature_norm", axis=channel_axis
    )(x)
    x = keras.layers.ReLU(name="low_level_feature_relu")(x)
    return x
