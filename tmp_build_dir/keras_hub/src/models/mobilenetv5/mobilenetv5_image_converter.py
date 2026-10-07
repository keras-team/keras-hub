from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.mobilenetv5.mobilenetv5_backbone import (
    MobileNetV5Backbone,
)


@keras_hub_export("keras_hub.layers.MobileNetV5ImageConverter")
class MobileNetV5ImageConverter(ImageConverter):
    backbone_cls = MobileNetV5Backbone
