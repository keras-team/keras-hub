from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.diffbin.diffbin_backbone import DiffBinBackbone
from keras_hub.src.models.diffbin.diffbin_image_converter import (
    DiffBinImageConverter,
)
from keras_hub.src.models.image_text_detector_preprocessor import (
    ImageTextDetectorPreprocessor,
)

@keras_hub_export("keras_hub.models.DiffBinPreprocessor")
class DiffBinPreprocessor(ImageTextDetectorPreprocessor):
    backbone_cls = DiffBinBackbone
    image_converter_cls = DiffBinImageConverter
