from keras_hub.src.models.opt.opt_backbone import OPTBackbone
from keras_hub.src.models.opt.opt_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, OPTBackbone)
