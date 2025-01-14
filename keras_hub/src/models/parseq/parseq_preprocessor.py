from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_segmenter_preprocessor import (
    ImageSegmenterPreprocessor,
)
from keras_hub.src.models.parseq.parseq_backbone import PARSeqBackbone
from keras_hub.src.models.parseq.parseq_image_converter import (
    PARSeqImageConverter,
)


@keras_hub_export("keras_hub.models.PARSeqPreprocessor")
class PARSeqPreprocessor(ImageSegmenterPreprocessor):
    backbone_cls = PARSeqBackbone
    image_converter_cls = PARSeqImageConverter
