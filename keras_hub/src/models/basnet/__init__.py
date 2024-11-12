from keras_hub.src.models.basnet.basnet import BASNet
from keras_hub.src.models.basnet.basnet_presets import basnet_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(basnet_presets, BASNet)
