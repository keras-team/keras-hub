from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.swin_transformer.swin_transformer_backbone import (
    SwinTransformerBackbone,
)


@keras_hub_export("keras_hub.layers.SwinTransformerImageConverter")
class SwinTransformerImageConverter(ImageConverter):
    backbone_cls = SwinTransformerBackbone
