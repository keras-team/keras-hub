from keras_hub.src.models.mpnet.mpnet_backbone import MPNetBackbone
from keras_hub.src.models.mpnet.mpnet_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, MPNetBackbone)
