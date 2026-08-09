from keras_hub.src.models.cspnet.cspnet_backbone import CSPNetBackbone
from keras_hub.src.models.cspnet.cspnet_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, CSPNetBackbone)
