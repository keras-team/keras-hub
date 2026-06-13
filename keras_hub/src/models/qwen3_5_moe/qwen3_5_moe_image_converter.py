from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen3_5.qwen3_5_image_converter import (
    Qwen3_5ImageConverter,
)
from keras_hub.src.models.qwen3_5_moe.qwen3_5_moe_backbone import (
    Qwen3_5MoeBackbone,
)


@keras_hub_export("keras_hub.models.Qwen3_5MoeImageConverter")
class Qwen3_5MoeImageConverter(Qwen3_5ImageConverter):
    """Image pre-processor for Qwen3.5-MoE-VL.

    Inherits all image preprocessing logic from the Qwen3.5 dense
    image converter (smart resize, patch extraction, grid_thw).
    """

    backbone_cls = Qwen3_5MoeBackbone
