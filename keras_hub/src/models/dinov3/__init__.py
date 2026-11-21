from keras_hub.src.models.dinov3.dinov3_backbone import DINOV3Backbone
from keras_hub.src.models.dinov3.dinov3_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, DINOV3Backbone)
