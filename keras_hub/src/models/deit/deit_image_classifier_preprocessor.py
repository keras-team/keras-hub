from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_classifier_preprocessor import (
    ImageClassifierPreprocessor,
)
from keras_hub.src.models.deit.deit_backbone import DeiTBackbone
from keras_hub.src.models.deit.deit_image_converter import DeiTImageConverter


@keras_hub_export("keras_hub.models.ViTImageClassifierPreprocessor")
class DeiTImageClassifierPreprocessor(ImageClassifierPreprocessor):
    backbone_cls = DeiTBackbone
    image_converter_cls = DeiTImageConverter
