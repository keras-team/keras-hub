from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.resnet.resnet_backbone import ResNetBackbone


@keras_hub_export("keras_hub.layers.ResNetImageConverter")
class ResNetImageConverter(ImageConverter):
    backbone_cls = ResNetBackbone
