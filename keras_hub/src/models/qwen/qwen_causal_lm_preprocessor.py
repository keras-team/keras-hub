from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.qwen.qwen_backbone import QwenBackbone
from keras_hub.src.models.qwen.qwen_tokenizer import QwenTokenizer


class QwenCausalLMPreprocessor(CausalLMPreprocessor):
    backbone_cls = QwenBackbone
    tokenizer_cls = QwenTokenizer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
