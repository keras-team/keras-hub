from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen3_5.qwen3_5_causal_lm_preprocessor import (
    Qwen3_5CausalLMPreprocessor,
)
from keras_hub.src.models.qwen3_5_moe.qwen3_5_moe_backbone import (
    Qwen3_5MoeBackbone,
)
from keras_hub.src.models.qwen3_5_moe.qwen3_5_moe_tokenizer import (
    Qwen3_5MoeTokenizer,
)


@keras_hub_export("keras_hub.models.Qwen3_5MoeCausalLMPreprocessor")
class Qwen3_5MoeCausalLMPreprocessor(Qwen3_5CausalLMPreprocessor):
    """Qwen3.5 MoE Causal LM preprocessor with multimodal support.

    Inherits all multimodal preprocessing logic from the Qwen3.5 dense
    preprocessor (image/video conversion, M-RoPE position IDs, vision
    indices, special token handling). The only differences are the
    ``backbone_cls`` and ``tokenizer_cls`` references.
    """

    backbone_cls = Qwen3_5MoeBackbone
    tokenizer_cls = Qwen3_5MoeTokenizer
