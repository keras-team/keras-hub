from keras_hub.src.models.swin_transformer.swin_transformer_backbone import (
    SwinTransformerBackbone,
)
from keras_hub.src.models.swin_transformer.swin_transformer_image_classifier import (
    SwinTransformerImageClassifier,
)
from keras_hub.src.models.swin_transformer.swin_transformer_image_classifier_preprocessor import (
    SwinTransformerImageClassifierPreprocessor,
)
from keras_hub.src.models.swin_transformer.swin_transformer_image_converter import (
    SwinTransformerImageConverter,
)
from keras_hub.src.models.swin_transformer.swin_transformer_presets import (
    backbone_presets,
)
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, SwinTransformerBackbone)
