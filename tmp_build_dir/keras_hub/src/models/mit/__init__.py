from keras_hub.src.models.mit.mit_backbone import MiTBackbone
from keras_hub.src.models.mit.mit_image_classifier import MiTImageClassifier
from keras_hub.src.models.mit.mit_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, MiTBackbone)
