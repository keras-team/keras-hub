from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.cspnet.cspnet_backbone import CSPNetBackbone
from keras_hub.src.models.cspnet.cspnet_image_classifier_preprocessor import (
    CSPNetImageClassifierPreprocessor,
)
from keras_hub.src.models.image_classifier import ImageClassifier


@keras_hub_export("keras_hub.models.CSPNetImageClassifier")
class CSPNetImageClassifier(ImageClassifier):
    backbone_cls = CSPNetBackbone
    preprocessor_cls = CSPNetImageClassifierPreprocessor
