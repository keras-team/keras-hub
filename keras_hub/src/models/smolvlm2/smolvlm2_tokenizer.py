"""Tokenizer for SmolVLM2 models."""

import re

from tokenizers import pre_tokenizers

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.smolvlm2.smolvlm2_backbone import SmolVLM2Backbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer

# `<row_R_col_C>` tags marking each crop of a split image.
ROW_COL_PATTERN = re.compile(r"<row_\d+_col_\d+>")


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
        "smolvlm2_2.2b_instruct"
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

    def set_vocabulary_and_merges(self, vocabulary, merges):
        super().set_vocabulary_and_merges(vocabulary, merges)
        # Rebuilt on every vocabulary change, including `load_assets()`. The
        # parent constructor calls this method, so the map always exists.
        self._row_col_token_ids = {
            token: token_id
            for token, token_id in (self.vocabulary or {}).items()
            if ROW_COL_PATTERN.fullmatch(token)
        }

    def _set_vocabulary_and_merges_tokenizers(self, vocabulary, merges):
        super()._set_vocabulary_and_merges_tokenizers(vocabulary, merges)
        # SmolLM2 pre-tokenizes with isolated digits and the GPT-2 byte-level
        # regex, as HF does. The base class's Llama3 split keeps "\n\n" before
        # a word as one piece, one token fewer per blank line than HF.
        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Digits(individual_digits=True),
                pre_tokenizers.ByteLevel(
                    add_prefix_space=self.add_prefix_space, use_regex=True
                ),
            ]
        )

    @property
    def row_col_token_ids(self):
        """`{"<row_R_col_C>": id}` for every crop tag in the vocabulary."""
        return dict(self._row_col_token_ids)

    @property
    def special_token_ids(self):
        # The crop tags are not registered as special tokens, since a
        # vocabulary without them would then fail to load. They still have
        # to be stripped from generated text.
        ids = super().special_token_ids
        return ids + list(self._row_col_token_ids.values())
