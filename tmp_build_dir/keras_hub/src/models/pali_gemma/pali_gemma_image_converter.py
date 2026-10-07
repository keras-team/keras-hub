from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.pali_gemma.pali_gemma_backbone import (
    PaliGemmaBackbone,
)


@keras_hub_export("keras_hub.layers.PaliGemmaImageConverter")
class PaliGemmaImageConverter(ImageConverter):
    backbone_cls = PaliGemmaBackbone
