from keras_hub.src.models.electra.electra_backbone import ElectraBackbone
from keras_hub.src.models.electra.electra_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, ElectraBackbone)
