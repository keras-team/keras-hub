from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.resizing_image_converter import (
    ResizingImageConverter,
)
from keras_hub.src.models.deeplab_v3.deeplab_v3_backbone import (
    DeepLabV3Backbone,
)


@keras_hub_export("keras_hub.layers.DeepLabV3ImageConverter")
class DeepLabV3ImageConverter(ResizingImageConverter):
    backbone_cls = DeepLabV3Backbone
