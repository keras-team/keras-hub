from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.xception.xception_backbone import XceptionBackbone
from keras_hub.src.models.xception.xception_image_classifier_preprocessor import (  # noqa: E501
    XceptionImageClassifierPreprocessor,
)


@keras_hub_export("keras_hub.models.XceptionImageClassifier")
class XceptionImageClassifier(ImageClassifier):
    backbone_cls = XceptionBackbone
    preprocessor_cls = XceptionImageClassifierPreprocessor
