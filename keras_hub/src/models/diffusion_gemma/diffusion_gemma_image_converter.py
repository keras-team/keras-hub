from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.diffusion_gemma.diffusion_gemma_backbone import (
    DiffusionGemmaBackbone,
)
from keras_hub.src.models.gemma4.gemma4_image_converter import (
    Gemma4ImageConverter,
)


@keras_hub_export("keras_hub.layers.DiffusionGemmaImageConverter")
class DiffusionGemmaImageConverter(Gemma4ImageConverter):
    backbone_cls = DiffusionGemmaBackbone
