from keras_hub.src.models.falcon.falcon_backbone import FalconBackbone
from keras_hub.src.models.falcon.falcon_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, FalconBackbone)
