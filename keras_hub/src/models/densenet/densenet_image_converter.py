from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.resizing_image_converter import (
    ResizingImageConverter,
)
from keras_hub.src.models.densenet.densenet_backbone import DenseNetBackbone


@keras_hub_export("keras_hub.layers.DenseNetImageConverter")
class DenseNetImageConverter(ResizingImageConverter):
    backbone_cls = DenseNetBackbone
