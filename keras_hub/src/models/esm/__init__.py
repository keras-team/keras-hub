from keras_hub.src.models.esm.esm_backbone import ESMBackbone
from keras_hub.src.models.esm.esm_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, ESMBackbone)
