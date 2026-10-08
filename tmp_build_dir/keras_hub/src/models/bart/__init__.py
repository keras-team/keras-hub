from keras_hub.src.models.bart.bart_backbone import BartBackbone
from keras_hub.src.models.bart.bart_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, BartBackbone)
