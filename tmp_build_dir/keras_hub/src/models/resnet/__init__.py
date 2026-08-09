from keras_hub.src.models.resnet.resnet_backbone import ResNetBackbone
from keras_hub.src.models.resnet.resnet_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, ResNetBackbone)
