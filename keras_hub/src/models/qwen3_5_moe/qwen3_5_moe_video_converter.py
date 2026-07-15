from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen3_5.qwen3_5_video_converter import (
    Qwen3_5VideoConverter,
)
from keras_hub.src.models.qwen3_5_moe.qwen3_5_moe_backbone import (
    Qwen3_5MoeBackbone,
)


@keras_hub_export("keras_hub.models.Qwen3_5MoeVideoConverter")
class Qwen3_5MoeVideoConverter(Qwen3_5VideoConverter):
    """Video pre-processor for Qwen3.5-MoE-VL.

    Inherits all video preprocessing logic from the Qwen3.5 dense
    video converter (smart resize, temporal padding, patch extraction).
    """

    backbone_cls = Qwen3_5MoeBackbone
