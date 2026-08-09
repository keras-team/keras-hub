from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.densenet.densenet_backbone import DenseNetBackbone
from keras_hub.src.models.densenet.densenet_image_converter import (
    DenseNetImageConverter,
)
from keras_hub.src.models.image_classifier_preprocessor import (
    ImageClassifierPreprocessor,
)


@keras_hub_export("keras_hub.models.DenseNetImageClassifierPreprocessor")
class DenseNetImageClassifierPreprocessor(ImageClassifierPreprocessor):
    backbone_cls = DenseNetBackbone
    image_converter_cls = DenseNetImageConverter
