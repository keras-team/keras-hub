from keras_hub.src.models.tipsv2.tipsv2_backbone import (  # noqa: E501
    TIPSv2Backbone,
)
from keras_hub.src.models.tipsv2.tipsv2_presets import (  # noqa: E501
    backbone_presets,
)
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, TIPSv2Backbone)
