from keras_hub.src.models.rwkv7.rwkv7_backbone import RWKV7Backbone
from keras_hub.src.models.rwkv7.rwkv7_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, RWKV7Backbone)
