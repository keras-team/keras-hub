from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.sam.sam_backbone import SAMBackbone


@keras_hub_export("keras_hub.layers.SAMImageConverter")
class SAMImageConverter(ImageConverter):
    backbone_cls = SAMBackbone
