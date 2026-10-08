from keras_hub.src.models.deit.deit_backbone import DeiTBackbone
from keras_hub.src.models.deit.deit_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, DeiTBackbone)
