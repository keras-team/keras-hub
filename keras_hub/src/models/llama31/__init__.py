from keras_hub.src.models.llama31.llama31_backbone import Llama31Backbone
from keras_hub.src.models.llama31.llama31_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, Llama31Backbone)
