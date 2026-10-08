from keras_hub.src.models.llama.llama_backbone import LlamaBackbone
from keras_hub.src.models.llama.llama_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, LlamaBackbone)
