from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.qwen3_moe.qwen3_moe_backbone import Qwen3MoeBackbone
from keras_hub.src.models.qwen3_moe.qwen3_moe_tokenizer import Qwen3MoeTokenizer


@keras_hub_export(
    "keras_hub.models.Qwen3MoeCausalLMPreprocessor",
)
class Qwen3MoeCausalLMPreprocessor(CausalLMPreprocessor):
    backbone_cls = Qwen3MoeBackbone
    tokenizer_cls = Qwen3MoeTokenizer
