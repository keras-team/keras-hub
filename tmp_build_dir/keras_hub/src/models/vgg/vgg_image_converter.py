from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.vgg.vgg_backbone import VGGBackbone


@keras_hub_export("keras_hub.layers.VGGImageConverter")
class VGGImageConverter(ImageConverter):
    backbone_cls = VGGBackbone
