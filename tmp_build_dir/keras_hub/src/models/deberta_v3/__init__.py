from keras_hub.src.models.deberta_v3.deberta_v3_backbone import (
    DebertaV3Backbone,
)
from keras_hub.src.models.deberta_v3.deberta_v3_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, DebertaV3Backbone)
