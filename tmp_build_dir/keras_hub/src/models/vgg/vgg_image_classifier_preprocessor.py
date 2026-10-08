from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_classifier_preprocessor import (
    ImageClassifierPreprocessor,
)
from keras_hub.src.models.vgg.vgg_backbone import VGGBackbone
from keras_hub.src.models.vgg.vgg_image_converter import VGGImageConverter


@keras_hub_export("keras_hub.models.VGGImageClassifierPreprocessor")
class VGGImageClassifierPreprocessor(ImageClassifierPreprocessor):
    backbone_cls = VGGBackbone
    image_converter_cls = VGGImageConverter
