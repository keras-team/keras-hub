from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.depth_anything.depth_anything_backbone import (
    DepthAnythingBackbone,
)


@keras_hub_export("keras_hub.layers.DepthAnythingImageConverter")
class DepthAnythingImageConverter(ImageConverter):
    backbone_cls = DepthAnythingBackbone
