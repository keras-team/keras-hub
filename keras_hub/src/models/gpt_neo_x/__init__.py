from keras_hub.src.models.gpt_neo_x.gpt_neo_x_backbone import GPTNeoXBackbone
from keras_hub.src.models.gpt_neo_x.gpt_neo_x_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, GPTNeoXBackbone)
