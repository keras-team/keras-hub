from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.qwen.qwen_backbone import Qwen2Backbone
from keras_hub.src.models.qwen.qwen_tokenizer import Qwen2Tokenizer


@keras_hub_export("keras_hub.models.Qwen2CausalLMPreprocessor")
class Qwen2CausalLMPreprocessor(CausalLMPreprocessor):
    backbone_cls = Qwen2Backbone
    tokenizer_cls = Qwen2Tokenizer
