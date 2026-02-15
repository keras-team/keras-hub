from keras_hub.src.models.metaclip_2.metaclip_2_backbone import (
    MetaCLIP2Backbone,
)
from keras_hub.src.models.metaclip_2.metaclip_2_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, MetaCLIP2Backbone)
