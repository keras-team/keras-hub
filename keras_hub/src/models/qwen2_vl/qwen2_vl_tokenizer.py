from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen.qwen_tokenizer import QwenTokenizer
from keras_hub.src.models.qwen2_vl.qwen2_vl_backbone import Qwen2VLBackbone

VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"
VISION_PAD_TOKEN = "<|vision_pad|>"
IMAGE_PAD_TOKEN = "<|image_pad|>"
VIDEO_PAD_TOKEN = "<|video_pad|>"


@keras_hub_export(
    [
        "keras_hub.tokenizers.Qwen2VLTokenizer",
        "keras_hub.models.Qwen2VLTokenizer",
    ]
)
class Qwen2VLTokenizer(QwenTokenizer):
    """Tokenizer for Qwen2-VL models.

    This tokenizer extends the base Qwen tokenizer with vision-related
    special tokens for multimodal input handling.

    Args:
        vocabulary: Dictionary mapping tokens to IDs, or path to file.
        merges: List of BPE merges, or path to merges file.
    """

    backbone_cls = Qwen2VLBackbone

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_vision_token_ids()

    def set_vocabulary_and_merges(self, vocabulary, merges):
        super().set_vocabulary_and_merges(vocabulary, merges)
        self._init_vision_token_ids()

    def _safe_token_to_id(self, token):
        if self.vocabulary is None:
            return None
        return self.vocabulary.get(token)

    def _init_vision_token_ids(self):
        # Multimodal token IDs used by preprocessing/model plumbing.
        self.image_token_id = self._safe_token_to_id(IMAGE_PAD_TOKEN)
        self.video_token_id = self._safe_token_to_id(VIDEO_PAD_TOKEN)
        self.vision_start_token_id = self._safe_token_to_id(
            VISION_START_TOKEN
        )
        self.vision_end_token_id = self._safe_token_to_id(VISION_END_TOKEN)
        self.vision_pad_token_id = self._safe_token_to_id(VISION_PAD_TOKEN)
        # Common alias names.
        self.image_pad_token_id = self.image_token_id
        self.video_pad_token_id = self.video_token_id
