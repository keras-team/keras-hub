from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.qwen3_5.qwen3_5_backbone import Qwen3_5Backbone
from keras_hub.src.models.qwen3_5.qwen3_5_tokenizer import Qwen3_5Tokenizer


@keras_hub_export("keras_hub.models.Qwen3_5CausalLMPreprocessor")
class Qwen3_5CausalLMPreprocessor(CausalLMPreprocessor):
    backbone_cls = Qwen3_5Backbone
    tokenizer_cls = Qwen3_5Tokenizer
