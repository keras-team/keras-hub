from keras_hub.src.models.bloom.bloom_backbone import BloomBackbone
from keras_hub.src.models.bloom.bloom_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, BloomBackbone)
