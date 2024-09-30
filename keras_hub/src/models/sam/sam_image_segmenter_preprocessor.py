from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_segmenter_preprocessor import (
    ImageSegmenterPreprocessor,
)
from keras_hub.src.models.sam.sam_backbone import SAMBackbone
from keras_hub.src.models.sam.sam_image_converter import SAMImageConverter


@keras_hub_export("keras_hub.models.SamImageSegmenterPreprocessor")
class SAMImageSegmenterPreprocessor(ImageSegmenterPreprocessor):
    backbone_cls = SAMBackbone
    image_converter_cls = SAMImageConverter
