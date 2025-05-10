from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen3.qwen3_backbone import Qwen3Backbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export(
    "keras_hub.models.Qwen3Tokenizer",
)
class Qwen3Tokenizer(BytePairTokenizer):
    """Tokenizer for Qwen3 models.

    This tokenizer implements byte-pair encoding (BPE) for Qwen3 models,
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

    backbone_cls = Qwen3Backbone

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
