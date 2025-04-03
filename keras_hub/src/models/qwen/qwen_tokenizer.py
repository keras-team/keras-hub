from keras_hub.src.models.qwen.qwen_backbone import QwenBackbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


class QwenTokenizer(BytePairTokenizer):
    """Tokenizer for Qwen models.

    This tokenizer implements byte-pair encoding (BPE) for Qwen models,
    handling special tokens like BOS (beginning of sequence) and EOS (end of
    sequence).

    Args:
        vocabulary: Dictionary mapping tokens to token IDs, or path to
            vocabulary file.
        merges: List of BPE merges, or path to merges file.
        bos_token: Beginning of sequence token. Defaults to None.
        eos_token: End of sequence token. Defaults to "<|endoftext|>".
        misc_special_tokens: Set of additional special tokens. Defaults to
            empty set.
    """

    backbone_cls = QwenBackbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        **kwargs,
    ):
        # Add EOS token
        eos_token = "<|endoftext|>"
        self._add_special_token(eos_token, "end_token")

        self.start_token_id = None
        self.start_token = None
        self.pad_token_id = 0

        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )
