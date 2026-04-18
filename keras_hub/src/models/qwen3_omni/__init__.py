from keras_hub.src.models.qwen3_omni.qwen3_omni_backbone import (
    Qwen3OmniBackbone,
)
from keras_hub.src.models.qwen3_omni.qwen3_omni_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, Qwen3OmniBackbone)
