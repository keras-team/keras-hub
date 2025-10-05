from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen3_omni_moe.qwen3_omni_moe_backbone import Qwen3OmniMoeBackbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export(
    "keras_hub.tokenizers.Qwen3OmniMoeTokenizer",
)
class Qwen3OmniMoeTokenizer(BytePairTokenizer):
    """Tokenizer for Qwen3-Omni MoE model.

    This tokenizer implements byte-pair encoding (BPE) for Qwen3-Omni MoE models,
    handling special tokens like BOS (beginning of sequence) and EOS (end of
    sequence). It supports multimodal capabilities including text, audio, and vision.

    Args:
        vocabulary: Dictionary mapping tokens to token IDs, or path to
            vocabulary file.
        merges: List of BPE merges, or path to merges file.
        bos_token: Beginning of sequence token. Defaults to None.
        eos_token: End of sequence token. Defaults to "<|im_end|>".
        misc_special_tokens: Set of additional special tokens. Defaults to
            empty set.

    Example:
    ```python
    # Create tokenizer
    tokenizer = Qwen3OmniMoeTokenizer.from_preset("qwen3_omni_moe_7b")
    
    # Tokenize text
    tokens = tokenizer("Hello, world!")
    # Returns: {'token_ids': array([[151644, 8948, 77091, 151645, 0, 0]]), 'padding_mask': array([[1, 1, 1, 1, 0, 0]])}
    ```
    """

    backbone_cls = Qwen3OmniMoeBackbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        **kwargs,
    ):
        # Add EOS token
        eos_token = "<|im_end|>"
        self._add_special_token(eos_token, "end_token")

        pad_token = "<|endoftext|>"
        self._add_special_token(pad_token, "pad_token")

        self.start_token_id = None
        self.start_token = None

        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )
