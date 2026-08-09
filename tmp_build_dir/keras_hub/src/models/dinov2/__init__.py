from keras_hub.src.models.dinov2.dinov2_backbone import DINOV2Backbone
from keras_hub.src.models.dinov2.dinov2_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, DINOV2Backbone)
