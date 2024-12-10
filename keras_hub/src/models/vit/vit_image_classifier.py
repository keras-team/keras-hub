from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.vit.vit_backbone import ViTBackbone
from keras_hub.src.models.vit.vit_image_classifier_preprocessor import (
    ViTImageClassifierPreprocessor,
)


@keras_hub_export("keras_hub.models.ViTImageClassifier")
class ViTImageClassifier(ImageClassifier):
    backbone_cls = ViTBackbone
    preprocessor_cls = ViTImageClassifierPreprocessor

    def __init__(self, pooling="first", **kwargs):
        super().__init__(pooling=pooling, **kwargs)
