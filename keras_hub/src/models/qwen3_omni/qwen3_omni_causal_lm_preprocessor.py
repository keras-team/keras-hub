from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.qwen3_omni.qwen3_omni_backbone import Qwen3OmniBackbone
from keras_hub.src.models.qwen3_omni.qwen3_omni_tokenizer import Qwen3OmniTokenizer


@keras_hub_export(
    "keras_hub.models.Qwen3OmniCausalLMPreprocessor",
)
class Qwen3OmniCausalLMPreprocessor(CausalLMPreprocessor):
    """Preprocessor for Qwen3-Omni causal language modeling.

    This preprocessor handles text tokenization for Qwen3-Omni models.

    Args:
        tokenizer: A `Qwen3OmniTokenizer` instance.
        sequence_length: int. The maximum sequence length.
        add_start_token: bool. Whether to add start token. Defaults to False.
        add_end_token: bool. Whether to add end token. Defaults to True.
    """

    backbone_cls = Qwen3OmniBackbone
    tokenizer_cls = Qwen3OmniTokenizer
