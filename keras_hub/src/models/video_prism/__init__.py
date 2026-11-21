from keras_hub.src.models.video_prism.video_prism_backbone import (
    VideoPrismBackbone,
)
from keras_hub.src.models.video_prism.video_prism_presets import (
    backbone_presets,
)
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, VideoPrismBackbone)
