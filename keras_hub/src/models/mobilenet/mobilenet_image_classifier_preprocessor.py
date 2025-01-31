from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_classifier_preprocessor import (
    ImageClassifierPreprocessor,
)
from keras_hub.src.models.mobilenet.mobilenet_backbone import MobileNetBackbone
from keras_hub.src.models.mobilenet.mobilenet_image_converter import (
    MobileNetImageConverter,
)


@keras_hub_export("keras_hub.models.MobileNetImageClassifierPreprocessor")
class MobileNetImageClassifierPreprocessor(ImageClassifierPreprocessor):
    backbone_cls = MobileNetBackbone
    image_converter_cls = MobileNetImageConverter
