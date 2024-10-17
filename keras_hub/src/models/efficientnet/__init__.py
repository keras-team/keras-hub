from keras_hub.src.models.efficientnet.efficientnet_backbone import (
    EfficientNetBackbone,
)
from keras_hub.src.models.efficientnet.efficientnet_presets import (
    backbone_presets,
)
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, EfficientNetBackbone)
