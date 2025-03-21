from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.qwen.qwen_backbone import QwenBackbone
from keras_hub.src.models.qwen.qwen_tokenizer import QwenTokenizer


@keras_hub_export(
    [
        "keras_hub.models.QwenCausalLMPreprocessor",
        "keras_hub.models.Qwen2CausalLMPreprocessor",
    ]
)
class QwenCausalLMPreprocessor(CausalLMPreprocessor):
    backbone_cls = QwenBackbone
    tokenizer_cls = QwenTokenizer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
