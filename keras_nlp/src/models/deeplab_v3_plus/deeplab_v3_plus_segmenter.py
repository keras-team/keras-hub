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
from keras_nlp.src.models.deeplab_v3_plus.deeplab_v3_plus_layers import (
    SpatialPyramidPooling,
)
from keras_nlp.src.models.task import Task

@keras_nlp_export(
    [
        "keras_nlp.models.DeepLabV3Plus",
        "keras_nlp.models.segmentation.DeepLabV3Plus",
    ]
)
class DeepLabV3Plus(Task):
    """DeepLabV3+ architecture for semantic segmentation.

    This class implements a DeepLabV3+ architecture as described in 
    [Encoder-Decoder with Atrous Separable Convolution for Semantic Image 
    Segmentation](https://arxiv.org/abs/1802.02611)(ECCV 2018)
    and [Rethinking Atrous Convolution for Semantic Image Segmentation](
        https://arxiv.org/abs/1706.05587)(CVPR 2017)

    Args:
    backbone: `keras.Model`. The backbone network for the model that is
        used as a feature extractor for the DeepLabV3+ Encoder. Should
        either be a `keras_nlp.models.backbones.backbone.Backbone` or a
        `keras.Model` that implements the `pyramid_outputs`
        property with keys "P2", "P3" etc as values. A
        somewhat sensible backbone to use in many cases is the
        `keras_nlp.models.ResNetBackbone.from_preset("resnet_v2_50")`.
    num_classes: int, the number of classes for the detection model. Note
        that the `num_classes` contains the background class, and the
        classes from the data should be represented by integers with range
        [0, `num_classes`).
    projection_filters: int, number of filters in the convolution layer
        projecting low-level features from the `backbone`.
    low_level_feature_key: str, layer level to extract the feature from one of the
    key from the `backbone` `pyramid_outputs`
        property such as  "P2", "P3" etc.
    spatial_pyramid_pooling_key: str, layer level to extract and perform 
    `spatial_pyramid_pooling`, one of the key from the `backbone` `pyramid_outputs`
        property such as  "P4", "P5" etc.
    spatial_pyramid_pooling: (Optional) a `keras.layers.Layer`. Also known
        as Atrous Spatial Pyramid Pooling (ASPP). Performs spatial pooling
        on different spatial levels in the pyramid, with dilation. If
        provided, the feature map from the backbone is passed to it inside
        the DeepLabV3 Encoder, otherwise SpatialPyramidPooling layer is used.
    dialtion_rates: (Optional) A `list` of integers for parallel dilated conv.
        Applied only when Default `SpatialPyramidPooling` is used. Usually a
        sample choice of rates are [6, 12, 18].
    segmentation_head: (Optional) a `keras.layers.Layer`. If provided, the
        outputs of the DeepLabV3 encoder is passed to this layer and it
        should predict the segmentation mask based on feature from backbone
        and feature from decoder, otherwise a default DeepLabV3
        convolutional head is used.

    Example:
    ```python
    images = np.ones(shape=(1, 96, 96, 3))
    labels = np.zeros(shape=(1, 96, 96, 1))
    backbone = keras_nlp.models.ResNetBackbone.from_preset("resnet_v2_50")

    model = keras_hub.models.DeepLabV3Plus(
        backbone= backbone,
        num_classes=3,
        projection_filters=48,
        low_level_feature_key="P2",
        spatial_pyramid_pooling_key="P5",
    )

    # Evaluate model
    model(images)

    # Train model
    model.compile(
        optimizer="adam",
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(images, labels, epochs=3)
    ```
    """

    def __init__(
        self,
        backbone,
        num_classes,
        low_level_feature_key,
        spatial_pyramid_pooling_key,
        projection_filters=48,
        spatial_pyramid_pooling=None,
        dialtion_rates=None,
        segmentation_head=None,
        **kwargs,
    ):
        if not isinstance(backbone, keras.layers.Layer) or not isinstance(
            backbone, keras.Model
        ):
            raise ValueError(
                "Argument `backbone` must be a `keras.layers.Layer` instance "
                f" or `keras.Model`. Received instead "
                f"backbone={backbone} (of type {type(backbone)})."
            )
            
        # === Functional Model ===
        inputs = backbone.input
        
        if spatial_pyramid_pooling is None:
            spatial_pyramid_pooling = SpatialPyramidPooling(
                dilation_rates=dialtion_rates
                )
        spatial_backbone_features = backbone.pyramid_outputs[spatial_pyramid_pooling_key]   
        spp_outputs = spatial_pyramid_pooling(spatial_backbone_features)

        low_level_backbone_feature = backbone.pyramid_outputs[low_level_feature_key]
        low_level_projected_features = apply_low_level_feature_network(low_level_backbone_feature, projection_filters)

        encoder_outputs = keras.layers.UpSampling2D(
            size=(8, 8),
            interpolation="bilinear",
            name="encoder_output_upsampling",
        )(spp_outputs)

        combined_encoder_outputs = keras.layers.Concatenate(axis=-1)(
            [encoder_outputs, low_level_projected_features]
        )

        if segmentation_head is None:
            x = keras.layers.Conv2D(
            name="segmentation_head_conv",
            filters=256,
            kernel_size=1,
            padding="same",
            use_bias=False,
            )(combined_encoder_outputs)
            x = keras.layers.BatchNormalization(
            name="segmentation_head_norm"
            )(x)
            x = keras.layers.ReLU(name="segmentation_head_relu")(x)
            x = keras.layers.UpSampling2D(
            size=(4, 4), interpolation="bilinear"
            )(x)
            # Classification layer
            outputs = keras.layers.Conv2D(
            name="segmentation_output",
            filters=num_classes,
            kernel_size=1,
            use_bias=False,
            padding="same",
            # Force the dtype of the classification layer to float32
            # to avoid the NAN loss issue when used with mixed
            # precision API.
            dtype="float32",
            )(x)
        else:
            outputs = segmentation_head(combined_encoder_outputs)
            
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
 
         # === Config ===       
        self.num_classes = num_classes
        self.backbone = backbone
        self.spatial_pyramid_pooling = spatial_pyramid_pooling
        self.projection_filters = projection_filters
        self.segmentation_head = segmentation_head
        self.dialtion_rates = dialtion_rates
        self.low_level_feature_key = low_level_feature_key
        self.spatial_pyramid_pooling_key = spatial_pyramid_pooling_key

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "backbone": keras.saving.serialize_keras_object(self.backbone),
            "spatial_pyramid_pooling": keras.saving.serialize_keras_object(
                self.spatial_pyramid_pooling
            ),
            "projection_filters": self.projection_filters,
            "segmentation_head": keras.saving.serialize_keras_object(
                self.segmentation_head
            ),
            "dialtion_rates": self.dialtion_rates,
            "low_level_feature_key": self.low_level_feature_key,
            "spatial_pyramid_pooling_key": self.spatial_pyramid_pooling_key,
        }


    @classmethod
    def from_config(cls, config):
        if "backbone" in config and isinstance(config["backbone"], dict):
            config["backbone"] = keras.layers.deserialize(config["backbone"])
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

def apply_low_level_feature_network(input_tensor, projection_filters):
    x = keras.layers.Conv2D(
        name="low_level_feature_conv",
        filters=projection_filters,
        kernel_size=1,
        padding="same",
        use_bias=False
    )(input_tensor)
    
    x = keras.layers.BatchNormalization(name="low_level_feature_norm")(x)
    x = keras.layers.ReLU(name="low_level_feature_relu")(x)
    return x



