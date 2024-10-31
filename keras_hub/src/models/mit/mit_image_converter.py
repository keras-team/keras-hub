from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.mit import MiTBackbone


@keras_hub_export("keras_hub.layers.MiTImageConverter")
class MiTImageConverter(ImageConverter):
    backbone_cls = MiTBackbone
