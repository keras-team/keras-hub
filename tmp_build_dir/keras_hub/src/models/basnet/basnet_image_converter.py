from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.basnet.basnet_backbone import BASNetBackbone


@keras_hub_export("keras_hub.layers.BASNetImageConverter")
class BASNetImageConverter(ImageConverter):
    backbone_cls = BASNetBackbone
