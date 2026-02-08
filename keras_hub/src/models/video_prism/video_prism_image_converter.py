from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.video_prism.video_prism_backbone import (
    VideoPrismBackbone,
)


@keras_hub_export("keras_hub.layers.VideoPrismImageConverter")
class VideoPrismImageConverter(ImageConverter):
    backbone_cls = VideoPrismBackbone
