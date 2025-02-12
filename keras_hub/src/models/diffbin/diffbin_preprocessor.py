from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.diffbin.diffbin_backbone import DiffBinBackbone
from keras_hub.src.models.diffbin.diffbin_image_converter import (
    DiffBinImageConverter,
)
from keras_hub.src.models.image_segmenter_preprocessor import (
    ImageSegmenterPreprocessor,
)


@keras_hub_export("keras_hub.models.DiffBinPreprocessor")
class DiffBinPreprocessor(ImageSegmenterPreprocessor):
    backbone_cls = DiffBinBackbone
    image_converter_cls = DiffBinImageConverter