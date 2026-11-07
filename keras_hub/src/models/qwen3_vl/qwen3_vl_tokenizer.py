from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen3_vl.qwen3_vl_backbone import Qwen3_VLBackbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.Qwen3_VLTokenizer",
        "keras_hub.models.Qwen3_VLTokenizer",
    ]
)
class Qwen3_VLTokenizer(BytePairTokenizer):
    """Tokenizer for Qwen3-VL models.

    This tokenizer implements byte-pair encoding (BPE) for Qwen3-VL models.
    In addition to the text special tokens, it registers the vision
    placeholders (`<|vision_start|>`, `<|vision_end|>`, `<|image_pad|>`,
    `<|video_pad|>`). All of them must be present in the vocabulary.

    Args:
        vocabulary: Dictionary mapping tokens to token IDs, or path to
            vocabulary file.
        merges: List of BPE merges, or path to merges file.
    """

    backbone_cls = Qwen3_VLBackbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        **kwargs,
    ):
        self._add_special_token("<|im_end|>", "end_token")
        self._add_special_token("<|endoftext|>", "pad_token")
        self._add_special_token("<|im_start|>", "im_start_token")
        self._add_special_token("<|vision_start|>", "vision_start_token")
        self._add_special_token("<|vision_end|>", "vision_end_token")
        self._add_special_token("<|image_pad|>", "image_token")
        self._add_special_token("<|video_pad|>", "video_token")

        self.start_token_id = None
        self.start_token = None

        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )
