from keras_hub.src.models.bge.bge_backbone import BgeBackbone
from keras_hub.src.models.bge.bge_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, BgeBackbone)
