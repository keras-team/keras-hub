"""Image converter for TIPSv2 models."""

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.tipsv2.tipsv2_backbone import TIPSv2Backbone


@keras_hub_export("keras_hub.layers.TIPSv2ImageConverter")
class TIPSv2ImageConverter(ImageConverter):
    """TIPSv2 image converter.

    Resizes images and scales pixel values to [0, 1].
    Unlike CLIP/SigLIP, TIPSv2 does NOT apply ImageNet normalization.

    The default image size is 448x448 as used in all TIPSv2 variants.
    """

    backbone_cls = TIPSv2Backbone
