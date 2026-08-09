from keras_hub.src.models.f_net.f_net_backbone import FNetBackbone
from keras_hub.src.models.f_net.f_net_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, FNetBackbone)
