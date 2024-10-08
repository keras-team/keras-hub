from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.resizing_image_converter import (
    ResizingImageConverter,
)
from keras_hub.src.models.mix_transformer import MiTBackbone


@keras_hub_export("keras_hub.layers.MiTImageConverter")
class MiTImageConverter(ResizingImageConverter):
    backbone_cls = MiTBackbone
