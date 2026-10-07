from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.densenet.densenet_backbone import DenseNetBackbone
from keras_hub.src.models.densenet.densenet_image_classifier_preprocessor import (  # noqa: E501
    DenseNetImageClassifierPreprocessor,
)
from keras_hub.src.models.image_classifier import ImageClassifier


@keras_hub_export("keras_hub.models.DenseNetImageClassifier")
class DenseNetImageClassifier(ImageClassifier):
    backbone_cls = DenseNetBackbone
    preprocessor_cls = DenseNetImageClassifierPreprocessor
