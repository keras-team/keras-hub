from keras_hub.src.models.flux2.flux2_backbone import Flux2Backbone
from keras_hub.src.models.flux2.flux2_presets import presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(presets, Flux2Backbone)
