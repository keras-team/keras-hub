from keras_hub.src.models.siglip.siglip_backbone import SigLIPBackbone
from keras_hub.src.models.siglip.siglip_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, SigLIPBackbone)
