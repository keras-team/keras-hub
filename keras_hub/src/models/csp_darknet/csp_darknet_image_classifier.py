from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.csp_darknet.csp_darknet_backbone import (
    CSPDarkNetBackbone,
)
from keras_hub.src.models.image_classifier import ImageClassifier


@keras_hub_export("keras_hub.models.CSPDarkNetImageClassifier")
class CSPDarkNetImageClassifier(ImageClassifier):
    backbone_cls = CSPDarkNetBackbone
