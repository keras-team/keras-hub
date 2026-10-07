from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.cspnet.cspnet_backbone import CSPNetBackbone


@keras_hub_export("keras_hub.layers.CSPNetImageConverter")
class CSPNetImageConverter(ImageConverter):
    backbone_cls = CSPNetBackbone
