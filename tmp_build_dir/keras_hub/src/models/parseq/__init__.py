from keras_hub.src.models.parseq.parseq_backbone import PARSeqBackbone
from keras_hub.src.models.parseq.parseq_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, PARSeqBackbone)
