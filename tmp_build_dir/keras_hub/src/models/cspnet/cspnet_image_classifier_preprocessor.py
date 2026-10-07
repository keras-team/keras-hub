from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.cspnet.cspnet_backbone import CSPNetBackbone
from keras_hub.src.models.cspnet.cspnet_image_converter import (
    CSPNetImageConverter,
)
from keras_hub.src.models.image_classifier_preprocessor import (
    ImageClassifierPreprocessor,
)


@keras_hub_export("keras_hub.models.CSPNetImageClassifierPreprocessor")
class CSPNetImageClassifierPreprocessor(ImageClassifierPreprocessor):
    backbone_cls = CSPNetBackbone
    image_converter_cls = CSPNetImageConverter
