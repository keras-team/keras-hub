"""Tokenizer for HRM-Text."""

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.hrm_text.hrm_text_backbone import HrmTextBackbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.HrmTextTokenizer",
        "keras_hub.models.HrmTextTokenizer",
    ]
)
class HrmTextTokenizer(BytePairTokenizer):
    """Byte-pair tokenizer used by HRM-Text.

    The official checkpoint uses the Qwen2 BPE vocabulary with HRM-Text's
    start, end, and padding special tokens.
    """

    backbone_cls = HrmTextBackbone

    def __init__(self, vocabulary=None, merges=None, **kwargs):
        self._add_special_token("<|im_start|>", "start_token")
        self._add_special_token("<|box_end|>", "end_token")
        self._add_special_token("<|endoftext|>", "pad_token")
        super().__init__(vocabulary=vocabulary, merges=merges, **kwargs)
