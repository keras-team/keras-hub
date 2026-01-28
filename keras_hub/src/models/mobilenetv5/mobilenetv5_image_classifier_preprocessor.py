from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_classifier_preprocessor import (
    ImageClassifierPreprocessor,
)
from keras_hub.src.models.mobilenetv5.mobilenetv5_backbone import (
    MobileNetV5Backbone,
)
from keras_hub.src.models.mobilenetv5.mobilenetv5_image_converter import (
    MobileNetV5ImageConverter,
)


@keras_hub_export("keras_hub.models.MobileNetV5ImageClassifierPreprocessor")
class MobileNetV5ImageClassifierPreprocessor(ImageClassifierPreprocessor):
    backbone_cls = MobileNetV5Backbone
    image_converter_cls = MobileNetV5ImageConverter
