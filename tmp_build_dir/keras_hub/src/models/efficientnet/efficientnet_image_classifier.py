from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.efficientnet.efficientnet_backbone import (
    EfficientNetBackbone,
)
from keras_hub.src.models.efficientnet.efficientnet_image_classifier_preprocessor import (  # noqa: E501
    EfficientNetImageClassifierPreprocessor,
)
from keras_hub.src.models.image_classifier import ImageClassifier


@keras_hub_export("keras_hub.models.EfficientNetImageClassifier")
class EfficientNetImageClassifier(ImageClassifier):
    backbone_cls = EfficientNetBackbone
    preprocessor_cls = EfficientNetImageClassifierPreprocessor
