from keras_hub.src.models.llama3.llama3_backbone import Llama3Backbone
from keras_hub.src.models.llama3.llama3_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, Llama3Backbone)
