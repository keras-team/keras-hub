from keras_hub.src.models.video_swin.video_swin_backbone import (
    VideoSwinBackbone,
)
from keras_hub.src.models.video_swin.video_swin_backbone_presets import (
    backbone_presets,
)
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, VideoSwinBackbone)
