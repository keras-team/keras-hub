from keras_hub.src.models.inception.inception_backbone import InceptionBackbone
from keras_hub.src.models.inception.inception_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, InceptionBackbone)