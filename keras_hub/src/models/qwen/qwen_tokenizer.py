from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen.qwen_backbone import Qwen2Backbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.Qwen2Tokenizer",
        "keras_hub.models.Qwen2Tokenizer",
    ]
)
class Qwen2Tokenizer(BytePairTokenizer):
    """Tokenizer for Qwen2 models.

    This tokenizer implements byte-pair encoding (BPE) for Qwen2 models,
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

    backbone_cls = Qwen2Backbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        eos_token="<|endoftext|>",
        **kwargs,
    ):
        # Initialize special tokens set
        special_tokens = set()

        # Add EOS token
        self._add_special_token(eos_token, "end_token")
        special_tokens.add(eos_token)

        self.pad_token_id = 0

        # Only pass non-None special tokens to parent class
        kwargs["unsplittable_tokens"] = list(special_tokens)
        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )
