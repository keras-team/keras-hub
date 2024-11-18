from keras_hub.src.models.clip.clip_backbone import CLIPBackbone
from keras_hub.src.models.clip.clip_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, CLIPBackbone)
