from keras_hub.src.models.qwen.qwen_backbone import QwenBackbone
from keras_hub.src.models.qwen.qwen_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, QwenBackbone)
