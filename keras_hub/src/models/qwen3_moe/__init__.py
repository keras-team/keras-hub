from keras_hub.src.models.qwen3_moe.qwen3_moe_backbone import Qwen3MoeBackbone
from keras_hub.src.models.qwen3_moe.qwen3_moe_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, Qwen3MoeBackbone)
