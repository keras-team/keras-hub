from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_segmenter_preprocessor import (
    ImageSegmenterPreprocessor,
)
from keras_hub.src.models.parseq.parseq_backbone import ParseQBackbone
from keras_hub.src.models.parseq.parseq_image_converter import (
    ParseQImageConverter,
)


@keras_hub_export("keras_hub.models.ParseQPreprocessor")
class ParseQPreprocessor(ImageSegmenterPreprocessor):
    backbone_cls = ParseQBackbone
    image_converter_cls = ParseQImageConverter
