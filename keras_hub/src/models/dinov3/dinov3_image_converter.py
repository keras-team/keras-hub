from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.dinov3.dinov3_backbone import DINOV3Backbone


@keras_hub_export("keras_hub.layers.DINOV3ImageConverter")
class DINOV3ImageConverter(ImageConverter):
    backbone_cls = DINOV3Backbone
