from keras_hub.src.models.smolvlm2.smolvlm2_backbone import SmolVLM2Backbone
from keras_hub.src.models.smolvlm2.smolvlm2_presets import smolvlm2_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(smolvlm2_presets, SmolVLM2Backbone)
