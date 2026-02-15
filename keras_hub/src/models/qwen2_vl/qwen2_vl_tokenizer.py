"""Qwen2-VL Tokenizer.

Extends QwenTokenizer with vision-related special tokens for
multimodal (image + text) processing.
"""

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen2_vl.qwen2_vl_backbone import Qwen2VLBackbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer

VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"
VISION_PAD_TOKEN = "<|vision_pad|>"


@keras_hub_export(
    [
        "keras_hub.tokenizers.Qwen2VLTokenizer",
        "keras_hub.models.Qwen2VLTokenizer",
    ]
)
class Qwen2VLTokenizer(BytePairTokenizer):
    """Tokenizer for Qwen2-VL models.

    This tokenizer implements byte-pair encoding (BPE) for Qwen2-VL
    models, extending the base Qwen tokenizer with vision-related
    special tokens for multimodal input handling.

    Args:
        vocabulary: Dictionary mapping tokens to IDs, or path to file.
        merges: List of BPE merges, or path to merges file.
    """

    backbone_cls = Qwen2VLBackbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        **kwargs,
    ):
        # EOS token (same as base Qwen).
        eos_token = "<|endoftext|>"
        self._add_special_token(eos_token, "end_token")

        self.start_token_id = None
        self.start_token = None
        self.pad_token_id = 0

        # Vision special tokens.
        self._add_special_token(VISION_START_TOKEN, "vision_start_token")
        self._add_special_token(VISION_END_TOKEN, "vision_end_token")
        self._add_special_token(VISION_PAD_TOKEN, "vision_pad_token")

        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )
