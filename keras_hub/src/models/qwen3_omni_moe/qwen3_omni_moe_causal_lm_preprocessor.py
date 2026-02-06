from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.qwen3_omni_moe.qwen3_omni_moe_backbone import Qwen3OmniMoeBackbone
from keras_hub.src.models.qwen3_omni_moe.qwen3_omni_moe_tokenizer import Qwen3OmniMoeTokenizer


@keras_hub_export(
    "keras_hub.models.Qwen3OmniMoeCausalLMPreprocessor",
)
class Qwen3OmniMoeCausalLMPreprocessor(CausalLMPreprocessor):
    """Preprocessor for Qwen3-Omni MoE causal language model.

    This preprocessor handles tokenization and preprocessing for the Qwen3-Omni MoE
    model, supporting multimodal inputs including text, audio, and vision.

    Args:
        tokenizer: A `Qwen3OmniMoeTokenizer` instance.
        sequence_length: int. The length of the packed sequence.
        add_start_token: bool. Whether to add the start token. Defaults to True.
        add_end_token: bool. Whether to add the end token. Defaults to True.

    Example:
    ```python
    # Create preprocessor
    preprocessor = Qwen3OmniMoeCausalLMPreprocessor.from_preset("qwen3_omni_moe_7b")
    
    # Preprocess text
    preprocessed = preprocessor(["Hello, world!", "How are you?"])
    ```
    """

    backbone_cls = Qwen3OmniMoeBackbone
    tokenizer_cls = Qwen3OmniMoeTokenizer
