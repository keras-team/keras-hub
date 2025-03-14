from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.inception.inception_backbone import InceptionBackbone
from keras_hub.src.models.inception.inception_image_classifier_preprocessor import (
    InceptionImageClassifierPreprocessor,
)


@keras_hub_export("keras_hub.models.InceptionImageClassifier")
class InceptionImageClassifier(ImageClassifier):
    backbone_cls = InceptionBackbone
    preprocessor_cls = InceptionImageClassifierPreprocessor