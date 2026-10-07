from keras_hub.src.models.xlm_roberta.xlm_roberta_backbone import (
    XLMRobertaBackbone,
)
from keras_hub.src.models.xlm_roberta.xlm_roberta_presets import (
    backbone_presets,
)
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, XLMRobertaBackbone)
