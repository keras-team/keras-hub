from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.efficientnet.efficientnet_backbone import (
    EfficientNetBackbone,
)
from keras_hub.src.models.efficientnet.efficientnet_image_converter import (
    EfficientNetImageConverter,
)
from keras_hub.src.models.image_classifier_preprocessor import (
    ImageClassifierPreprocessor,
)


@keras_hub_export("keras_hub.models.EfficientNetImageClassifierPreprocessor")
class EfficientNetImageClassifierPreprocessor(ImageClassifierPreprocessor):
    backbone_cls = EfficientNetBackbone
    image_converter_cls = EfficientNetImageConverter
