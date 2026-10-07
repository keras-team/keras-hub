from keras_hub.src.models.densenet.densenet_backbone import DenseNetBackbone
from keras_hub.src.models.densenet.densenet_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, DenseNetBackbone)
