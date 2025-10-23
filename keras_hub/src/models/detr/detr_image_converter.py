from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.detr.detr_backbone import DETRBackbone


@keras_hub_export("keras_hub.layers.DETRImageConverter")
class DETRImageConverter(ImageConverter):
    backbone_cls = DETRBackbone
