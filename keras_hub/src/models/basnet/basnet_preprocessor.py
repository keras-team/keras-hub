from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.basnet.basnet_backbone import BASNetBackbone
from keras_hub.src.models.basnet.basnet_image_converter import (
    BASNetImageConverter,
)
from keras_hub.src.models.image_segmenter_preprocessor import (
    ImageSegmenterPreprocessor,
)


@keras_hub_export("keras_hub.models.BASNetPreprocessor")
class BASNetPreprocessor(ImageSegmenterPreprocessor):
    backbone_cls = BASNetBackbone
    image_converter_cls = BASNetImageConverter
