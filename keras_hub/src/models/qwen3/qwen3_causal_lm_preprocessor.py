from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.qwen3.qwen3_backbone import Qwen3Backbone
from keras_hub.src.models.qwen3.qwen3_tokenizer import Qwen3Tokenizer


@keras_hub_export("keras_hub.models.Qwen3CausalLMPreprocessor")
class Qwen3CausalLMPreprocessor(CausalLMPreprocessor):
    backbone_cls = Qwen3Backbone
    tokenizer_cls = Qwen3Tokenizer
