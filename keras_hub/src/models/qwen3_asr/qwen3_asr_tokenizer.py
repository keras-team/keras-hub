from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen3_asr.qwen3_asr_backbone import Qwen3ASRBackbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.Qwen3ASRTokenizer",
        "keras_hub.models.Qwen3ASRTokenizer",
    ]
)
class Qwen3ASRTokenizer(BytePairTokenizer):
    """Tokenizer for Qwen3-ASR models.

    This tokenizer uses byte-pair encoding (BPE) for the Qwen3-ASR speech
    recognition model. It shares the same vocabulary format as Qwen3 text
    models, with special tokens for end-of-sequence and padding.

    Args:
        vocabulary: Dictionary mapping tokens to token IDs, or path to
            vocabulary file.
        merges: List of BPE merges, or path to merges file.
    """

    backbone_cls = Qwen3ASRBackbone

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
