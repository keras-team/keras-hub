from keras_hub.src.models.sam3.sam3_backbone import SAM3Backbone
from keras_hub.src.models.sam3.sam3_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, SAM3Backbone)
