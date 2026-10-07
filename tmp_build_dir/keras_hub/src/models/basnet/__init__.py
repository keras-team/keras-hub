from keras_hub.src.models.basnet.basnet_backbone import BASNetBackbone
from keras_hub.src.models.basnet.basnet_presets import basnet_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(basnet_presets, BASNetBackbone)
