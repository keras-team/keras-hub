from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.deit.deit_backbone import DeiTBackbone


@keras_hub_export("keras_hub.layers.DeiTImageConverter")
class DeiTImageConverter(ImageConverter):
    backbone_cls = DeiTBackbone
