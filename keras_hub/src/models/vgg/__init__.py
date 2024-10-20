from keras_hub.src.models.vgg.vgg_backbone import VGGBackbone
from keras_hub.src.models.vgg.vgg_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, VGGBackbone)
