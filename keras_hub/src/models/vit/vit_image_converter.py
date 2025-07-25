from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.vit.vit_backbone import ViTBackbone


@keras_hub_export("keras_hub.layers.ViTImageConverter")
class ViTImageConverter(ImageConverter):
    backbone_cls = ViTBackbone
