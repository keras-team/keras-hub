from keras_hub.src.models.mix_transformer.mix_transformer_backbone import (
    MiTBackbone,
)
from keras_hub.src.models.mix_transformer.mix_transformer_classifier import (
    MiTImageClassifier,
)
from keras_hub.src.models.mix_transformer.mix_transformer_presets import (
    backbone_presets,
)
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, MiTBackbone)
