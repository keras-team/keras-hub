from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.qwen.qwen_tokenizer import QwenTokenizer
from keras_hub.src.models.qwen_moe.qwen_moe_backbone import QwenMoeBackbone


@keras_hub_export(
    [
        "keras_hub.models.QwenMoeCausalLMPreprocessor",
    ]
)
class QwenMoeCausalLMPreprocessor(CausalLMPreprocessor):
    backbone_cls = QwenMoeBackbone
    tokenizer_cls = QwenTokenizer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
