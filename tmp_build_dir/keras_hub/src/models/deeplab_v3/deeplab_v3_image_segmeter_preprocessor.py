from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.deeplab_v3.deeplab_v3_backbone import (
    DeepLabV3Backbone,
)
from keras_hub.src.models.deeplab_v3.deeplab_v3_image_converter import (
    DeepLabV3ImageConverter,
)
from keras_hub.src.models.image_segmenter_preprocessor import (
    ImageSegmenterPreprocessor,
)


@keras_hub_export("keras_hub.models.DeepLabV3ImageSegmenterPreprocessor")
class DeepLabV3ImageSegmenterPreprocessor(ImageSegmenterPreprocessor):
    backbone_cls = DeepLabV3Backbone
    image_converter_cls = DeepLabV3ImageConverter
