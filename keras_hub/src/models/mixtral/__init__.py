from keras_hub.src.models.mixtral.mixtral_backbone import MixtralBackbone
from keras_hub.src.models.mixtral.mixtral_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, MixtralBackbone)
