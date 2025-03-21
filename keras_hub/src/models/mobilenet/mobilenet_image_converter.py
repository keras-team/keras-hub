from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.mobilenet.mobilenet_backbone import MobileNetBackbone


@keras_hub_export("keras_hub.layers.MobileNetImageConverter")
class MobileNetImageConverter(ImageConverter):
    backbone_cls = MobileNetBackbone
