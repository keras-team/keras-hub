from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.densenet.densenet_backbone import DenseNetBackbone


@keras_hub_export("keras_hub.layers.DenseNetImageConverter")
class DenseNetImageConverter(ImageConverter):
    backbone_cls = DenseNetBackbone
