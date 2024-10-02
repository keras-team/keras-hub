from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.resizing_image_converter import (
    ResizingImageConverter,
)
from keras_hub.src.models.sam.sam_backbone import SAMBackbone


@keras_hub_export("keras_hub.layers.SAMImageConverter")
class SAMImageConverter(ResizingImageConverter):
    backbone_cls = SAMBackbone
