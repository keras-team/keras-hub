from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.hgnetv2.hgnetv2_backbone import HGNetV2Backbone
from keras_hub.src.models.hgnetv2.hgnetv2_image_converter import (
    HGNetV2ImageConverter,
)
from keras_hub.src.models.image_classifier_preprocessor import (
    ImageClassifierPreprocessor,
)


@keras_hub_export("keras_hub.models.HGNetV2ImageClassifierPreprocessor")
class HGNetV2ImageClassifierPreprocessor(ImageClassifierPreprocessor):
    backbone_cls = HGNetV2Backbone
    image_converter_cls = HGNetV2ImageConverter
