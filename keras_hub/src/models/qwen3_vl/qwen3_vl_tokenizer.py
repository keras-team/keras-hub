from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen3_vl.qwen3_vl_backbone import Qwen3VLBackbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer

# Marked `"special": true` in the HF `tokenizer_config.json`, so HF strips
# them with `skip_special_tokens=True`. `<think>` and `</think>` are not.
EXTRA_SPECIAL_TOKENS = (
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_pad|>",
)


@keras_hub_export(
    [
        "keras_hub.tokenizers.Qwen3VLTokenizer",
        "keras_hub.models.Qwen3VLTokenizer",
    ]
)
class Qwen3VLTokenizer(BytePairTokenizer):
    """Tokenizer for Qwen3-VL models.

    This tokenizer implements byte-pair encoding (BPE) for Qwen3-VL models.
    In addition to the text special tokens, it registers the vision
    placeholders (`<|vision_start|>`, `<|vision_end|>`, `<|image_pad|>`,
    `<|video_pad|>`). All of them must be present in the vocabulary.
    `<|endoftext|>` is also `end_token2`, the second end token of
    generation.

    Args:
        vocabulary: Dictionary mapping tokens to token IDs, or path to
            vocabulary file.
        merges: List of BPE merges, or path to merges file.

    Examples:
    ```python
    tokenizer = keras_hub.tokenizers.Qwen3VLTokenizer.from_preset(
        "hf://Qwen/Qwen3-VL-2B-Instruct"
    )
    tokenizer("The quick brown fox jumped.")
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))
    # The id of the image placeholder token.
    tokenizer.image_token_id
    ```
    """

    backbone_cls = Qwen3VLBackbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        **kwargs,
    ):
        self._add_special_token("<|im_end|>", "end_token")
        self._add_special_token("<|endoftext|>", "pad_token")
        # The second end token in the HF `generation_config.json`.
        self._add_special_token("<|endoftext|>", "end_token2")
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

    @property
    def special_token_ids(self):
        # The other HF special tokens are not registered, since a vocabulary
        # without them would then fail to load. They are still stripped from
        # generated text.
        ids = super().special_token_ids
        vocabulary = self.vocabulary or {}
        extra = [vocabulary[t] for t in EXTRA_SPECIAL_TOKENS if t in vocabulary]
        return ids + extra
