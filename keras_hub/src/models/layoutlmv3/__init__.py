"""LayoutLMv3 model."""

from keras_hub.src.models.layoutlmv3.layoutlmv3_backbone import LayoutLMv3Backbone
from keras_hub.src.models.layoutlmv3.layoutlmv3_tokenizer import LayoutLMv3Tokenizer
from keras_hub.src.models.layoutlmv3.document_classifier import LayoutLMv3DocumentClassifier
from keras_hub.src.models.layoutlmv3.document_classifier import LayoutLMv3DocumentClassifierPreprocessor
from keras_hub.src.models.layoutlmv3.layoutlmv3_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, LayoutLMv3Backbone) 