from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.mit.mit_backbone import MiTBackbone
from keras_hub.src.models.mit.mit_image_classifier_preprocessor import (
    MiTImageClassifierPreprocessor,
)


@keras_hub_export("keras_hub.models.MiTImageClassifier")
class MiTImageClassifier(ImageClassifier):
    backbone_cls = MiTBackbone
    preprocessor_cls = MiTImageClassifierPreprocessor
