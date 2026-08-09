from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_classifier_preprocessor import (
    ImageClassifierPreprocessor,
)
from keras_hub.src.models.xception.xception_backbone import XceptionBackbone
from keras_hub.src.models.xception.xception_image_converter import (
    XceptionImageConverter,
)


@keras_hub_export("keras_hub.models.XceptionImageClassifierPreprocessor")
class XceptionImageClassifierPreprocessor(ImageClassifierPreprocessor):
    backbone_cls = XceptionBackbone
    image_converter_cls = XceptionImageConverter
