from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_classifier_preprocessor import (
    ImageClassifierPreprocessor,
)
from keras_hub.src.models.resnet.resnet_backbone import ResNetBackbone
from keras_hub.src.models.resnet.resnet_image_converter import (
    ResNetImageConverter,
)


@keras_hub_export("keras_hub.models.ResNetImageClassifierPreprocessor")
class ResNetImageClassifierPreprocessor(ImageClassifierPreprocessor):
    backbone_cls = ResNetBackbone
    image_converter_cls = ResNetImageConverter
