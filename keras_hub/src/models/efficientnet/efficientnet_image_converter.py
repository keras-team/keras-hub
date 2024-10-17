from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.efficientnet.efficientnet_backbone import (
    EfficientNetBackbone,
)


@keras_hub_export("keras_hub.layers.EfficientNetImageConverter")
class EfficientNetImageConverter(ImageConverter):
    backbone_cls = EfficientNetBackbone
