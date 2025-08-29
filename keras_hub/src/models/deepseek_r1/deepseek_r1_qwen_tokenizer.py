from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen.qwen_backbone import QwenBackbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.DeepSeekR1QwenTokenizer",
        "keras_hub.tokenizers.DeepSeekR1Qwen2Tokenizer",
        "keras_hub.models.DeepSeekR1QwenTokenizer",
        "keras_hub.models.DeepSeekR1Qwen2Tokenizer",
    ]
)
class DeepSeekR1QwenTokenizer(BytePairTokenizer):
    """Tokenizer for DeepSeekR1-Distilled Qwen models.

    This tokenizer implements byte-pair encoding (BPE) for DeepSeekR1-Distilled
    Qwen models, handling special tokens like BOS (beginning of sequence)
    and EOS (end of sequence).

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
        eos_token = "<｜end▁of▁sentence｜>"
        self._add_special_token(eos_token, "eos_token")
        bos_token = "<｜begin▁of▁sentence｜>"
        self._add_special_token(bos_token, "bos_token")

        self.end_token_id = 151643
        self.start_token_id = 151646
        self.start_token = None
        self.pad_token_id = 0

        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )
