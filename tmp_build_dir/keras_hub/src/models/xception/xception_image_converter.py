from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.xception.xception_backbone import XceptionBackbone


@keras_hub_export("keras_hub.layers.XceptionImageConverter")
class XceptionImageConverter(ImageConverter):
    backbone_cls = XceptionBackbone
