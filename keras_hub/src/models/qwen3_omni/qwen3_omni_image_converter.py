from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.qwen3_omni.qwen3_omni_backbone import (
    Qwen3OmniBackbone,
)


@keras_hub_export("keras_hub.layers.Qwen3OmniImageConverter")
class Qwen3OmniImageConverter(ImageConverter):
    backbone_cls = Qwen3OmniBackbone
