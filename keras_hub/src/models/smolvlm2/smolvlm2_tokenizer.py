"""Tokenizer for SmolVLM2 models."""

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.smolvlm2.smolvlm2_backbone import SmolVLM2Backbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.SmolVLM2Tokenizer",
        "keras_hub.models.SmolVLM2Tokenizer",
    ]
)
class SmolVLM2Tokenizer(BytePairTokenizer):
    """Byte-pair tokenizer for SmolVLM2 models.

    This tokenizer implements GPT2-style byte-level BPE for
    SmolVLM2 models, with special tokens for multimodal inputs.

    This tokenizer does not handle special tokens or routing logic
    for multimodal inputs; use
    `keras_hub.models.SmolVLM2CausalLMPreprocessor` for that.

    Args:
        vocabulary: Dictionary mapping tokens to token IDs, or path to
            vocabulary file.
        merges: List of BPE merges, or path to merges file.

    Examples:
    ```python
    tokenizer = keras_hub.tokenizers.SmolVLM2Tokenizer.from_preset(
        "smolvlm2_2b_instruct"
    )
    tokenizer("Hello, world!")
    ```
    """

    backbone_cls = SmolVLM2Backbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        **kwargs,
    ):
        # End of sequence token.
        eos_token = "<|im_end|>"
        self._add_special_token(eos_token, "end_token")

        # Start of sequence / BOS token.
        bos_token = "<|im_start|>"
        self._add_special_token(bos_token, "start_token")

        # Image placeholder token.
        image_token = "<image>"
        self._add_special_token(image_token, "image_token")

        # End of utterance token — also acts as a second stop token
        # for generation (SmolVLM2 chat format ends assistant turns
        # with <end_of_utterance>, not <|im_end|>).
        eou_token = "<end_of_utterance>"
        self._add_special_token(eou_token, "end_of_utterance_token")
        self._add_special_token(eou_token, "end_token2")

        # Fake token around image (sentinel wrapping expanded image
        # sequences).
        fake_image_token = "<fake_token_around_image>"
        self._add_special_token(fake_image_token, "fake_image_token")

        # Global image token (marks the downscaled whole-image view).
        global_image_token = "<global-img>"
        self._add_special_token(global_image_token, "global_image_token")

        self.pad_token_id = 0

        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )
