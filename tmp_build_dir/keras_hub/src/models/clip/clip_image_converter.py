from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.clip.clip_backbone import CLIPBackbone


@keras_hub_export("keras_hub.layers.CLIPImageConverter")
class CLIPImageConverter(ImageConverter):
    backbone_cls = CLIPBackbone
