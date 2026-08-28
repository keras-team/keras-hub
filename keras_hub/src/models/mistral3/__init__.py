from keras_hub.src.models.mistral3.mistral3_backbone import Mistral3Backbone
from keras_hub.src.models.mistral3.mistral3_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, Mistral3Backbone)
