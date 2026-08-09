from keras_hub.src.models.roberta.roberta_backbone import RobertaBackbone
from keras_hub.src.models.roberta.roberta_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, RobertaBackbone)
