from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_segmenter_preprocessor import (
    ImageSegmenterPreprocessor,
)
from keras_hub.src.models.segformer.segformer_backbone import SegFormerBackbone
from keras_hub.src.models.segformer.segformer_image_converter import (
    SegFormerImageConverter,
)


@keras_hub_export("keras_hub.models.SegFormerImageSegmenterPreprocessor")
class SegFormerImageSegmenterPreprocessor(ImageSegmenterPreprocessor):
    backbone_cls = SegFormerBackbone
    image_converter_cls = SegFormerImageConverter
