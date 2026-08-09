from keras_hub.src.models.flux.flux_model import FluxBackbone
from keras_hub.src.models.flux.flux_presets import presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(presets, FluxBackbone)
