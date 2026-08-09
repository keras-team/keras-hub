from keras_hub.src.models.d_fine.d_fine_backbone import DFineBackbone
from keras_hub.src.models.d_fine.d_fine_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, DFineBackbone)
