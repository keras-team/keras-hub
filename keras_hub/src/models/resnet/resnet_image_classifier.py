from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.resnet.resnet_backbone import ResNetBackbone
from keras_hub.src.models.resnet.resnet_image_classifier_preprocessor import (
    ResNetImageClassifierPreprocessor,
)


@keras_hub_export("keras_hub.models.ResNetImageClassifier")
class ResNetImageClassifier(ImageClassifier):
    backbone_cls = ResNetBackbone
    preprocessor_cls = ResNetImageClassifierPreprocessor
