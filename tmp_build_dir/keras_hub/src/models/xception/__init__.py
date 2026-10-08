from keras_hub.src.models.xception.xception_backbone import XceptionBackbone
from keras_hub.src.models.xception.xception_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, XceptionBackbone)
