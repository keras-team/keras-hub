from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.parseq.parseq_backbone import PARSeqBackbone
from keras_hub.src.models.parseq.parseq_image_converter import (
    PARSeqImageConverter,
)
from keras_hub.src.models.text_recognition_preprocessor import (
    TextRecognitionPreprocessor,
)


@keras_hub_export("keras_hub.models.PARSeqPreprocessor")
class PARSeqPreprocessor(TextRecognitionPreprocessor):
    backbone_cls = PARSeqBackbone
    image_converter_cls = PARSeqImageConverter
