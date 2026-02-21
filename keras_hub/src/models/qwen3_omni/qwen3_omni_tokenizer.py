from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen3_omni.qwen3_omni_backbone import (
    Qwen3OmniBackbone,
)
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export(
    "keras_hub.tokenizers.Qwen3OmniTokenizer",
)
class Qwen3OmniTokenizer(BytePairTokenizer):
    """Tokenizer for Qwen3-Omni model.

    This tokenizer implements byte-pair encoding (BPE) for Qwen3-Omni models,
    handling special tokens like EOS (end of sequence) and PAD (padding).

    Args:
        vocabulary: Dictionary mapping tokens to token IDs, or path to
            vocabulary file.
        merges: List of BPE merges, or path to merges file.
        **kwargs: Additional keyword arguments passed to the parent
            `BytePairTokenizer` class.

    Examples:

    ```python
    # Load a preset tokenizer
    tokenizer = keras_hub.tokenizers.Qwen3OmniTokenizer.from_preset(
        "qwen3_omni_0.5b_en"
    )

    # Tokenize text
    tokenizer("The quick brown fox jumps.")
    ```
    """

    backbone_cls = Qwen3OmniBackbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        **kwargs,
    ):
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
