from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_classifier_preprocessor import (
    ImageClassifierPreprocessor,
)
from keras_hub.src.models.vit.vit_backbone import ViTBackbone
from keras_hub.src.models.vit.vit_image_converter import ViTImageConverter


@keras_hub_export("keras_hub.models.ViTImageClassifierPreprocessor")
class ViTImageClassifierPreprocessor(ImageClassifierPreprocessor):
    backbone_cls = ViTBackbone
    image_converter_cls = ViTImageConverter
