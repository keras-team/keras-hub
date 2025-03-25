from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.inception.image_classifier_preprocessor import (
    ImageClassifierPreprocessor,
)
from keras_hub.src.models.inception.inception_backbone import InceptionBackbone
from keras_hub.src.models.inception.inception_image_converter import (
    InceptionImageConverter,
)


@keras_hub_export("keras_hub.models.InceptionImageClassifierPreprocessor")
class InceptionImageClassifierPreprocessor(ImageClassifierPreprocessor):
    backbone_cls = InceptionBackbone
    image_converter_cls = InceptionImageConverter