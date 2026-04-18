from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen3_5_moe.qwen3_5_moe_backbone import (
    Qwen3_5MoeBackbone,
)
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.Qwen3_5MoeTokenizer",
        "keras_hub.models.Qwen3_5MoeTokenizer",
    ] 
)
class Qwen3_5MoeTokenizer(BytePairTokenizer):
    """Tokenizer for Qwen3.5 MoE models.

    This tokenizer implements byte-pair encoding (BPE) for Qwen3.5 MoE
    models.

    Args:
        vocabulary: Dictionary mapping tokens to token IDs, or path to
            vocabulary file.
        merges: List of BPE merges, or path to merges file.
        has_vision_tokens: bool. Whether to register vision-related
            special tokens. Default ``True``.
    """

    backbone_cls = Qwen3_5MoeBackbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        has_vision_tokens=True,
        **kwargs,
    ):
        self.has_vision_tokens = has_vision_tokens

        self._add_special_token("<|im_end|>", "end_token")
        self._add_special_token("<|endoftext|>", "pad_token")
        self._add_special_token("<|im_start|>", "im_start_token")

        self.start_token_id = None
        self.start_token = None

        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )

        if has_vision_tokens:
            self._add_special_token(
                "<|vision_start|>", "vision_start_token"
            )
            self._add_special_token(
                "<|vision_end|>", "vision_end_token"
            )
            self._add_special_token(
                "<|image_pad|>", "image_token"
            )
            self._add_special_token(
                "<|video_pad|>", "video_token"
            )
            if vocabulary is not None:
                self._update_special_token_ids()

    def get_config(self):
        config = super().get_config()
        config.update(
            {"has_vision_tokens": self.has_vision_tokens}
        )
        return config
