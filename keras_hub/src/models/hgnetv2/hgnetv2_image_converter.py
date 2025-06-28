from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.hgnetv2.hgnetv2_backbone import HGNetV2Backbone


@keras_hub_export("keras_hub.layers.HGNetV2ImageConverter")
class HGNetV2ImageConverter(ImageConverter):
    backbone_cls = HGNetV2Backbone
