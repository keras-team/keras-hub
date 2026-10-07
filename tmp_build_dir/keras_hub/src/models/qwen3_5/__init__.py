from keras_hub.src.models.qwen3_5.qwen3_5_backbone import Qwen3_5Backbone
from keras_hub.src.models.qwen3_5.qwen3_5_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, Qwen3_5Backbone)
