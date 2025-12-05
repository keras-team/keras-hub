from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.deepseek_r1.deepseek_r1_qwen_tokenizer import (
    DeepSeekR1QwenTokenizer,
)
from keras_hub.src.models.qwen.qwen_backbone import QwenBackbone


@keras_hub_export(
    [
        "keras_hub.models.DeepSeekR1QwenCausalLMPreprocessor",
        "keras_hub.models.DeepSeekR1Qwen2CausalLMPreprocessor",
    ]
)
class DeepSeekR1QwenCausalLMPreprocessor(CausalLMPreprocessor):
    backbone_cls = QwenBackbone
    tokenizer_cls = DeepSeekR1QwenTokenizer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
