from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.mobilenet.mobilenet_backbone import MobileNetBackbone
from keras_hub.src.models.mobilenet.mobilenet_image_classifier_preprocessor import (
    MobileNetImageClassifierPreprocessor,
)


@keras_hub_export("keras_hub.models.MobileNetImageClassifier")
class MobileNetImageClassifier(ImageClassifier):
    backbone_cls = MobileNetBackbone
    preprocessor_cls = MobileNetImageClassifierPreprocessor
