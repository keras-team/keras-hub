from keras_hub.src.models.t5.t5_backbone import T5Backbone
from keras_hub.src.models.t5.t5_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, T5Backbone)
