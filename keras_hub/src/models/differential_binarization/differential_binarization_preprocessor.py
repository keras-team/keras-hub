from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.differential_binarization.differential_binarization_backbone import (
    DifferentialBinarizationBackbone,
)
from keras_hub.src.models.differential_binarization.differential_binarization_image_converter import (
    DifferentialBinarizationImageConverter,
)
from keras_hub.src.models.image_segmenter_preprocessor import (
    ImageSegmenterPreprocessor,
)


@keras_hub_export("keras_hub.models.DifferentialBinarizationPreprocessor")
class DifferentialBinarizationPreprocessor(ImageSegmenterPreprocessor):
    backbone_cls = DifferentialBinarizationBackbone
    image_converter_cls = DifferentialBinarizationImageConverter
