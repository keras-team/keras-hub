from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen3_5.qwen3_5_backbone import Qwen3_5Backbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export("keras_hub.models.Qwen3_5Tokenizer")
class Qwen3_5Tokenizer(BytePairTokenizer):
    """Tokenizer for Qwen3.5 models.

    This tokenizer implements byte-pair encoding (BPE) for Qwen3.5 models.

    Args:
        vocabulary: Dictionary mapping tokens to token IDs, or path to
            vocabulary file.
        merges: List of BPE merges, or path to merges file.
    """

    backbone_cls = Qwen3_5Backbone

    def __init__(self, vocabulary=None, merges=None, **kwargs):
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
