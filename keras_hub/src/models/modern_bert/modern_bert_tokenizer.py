import re

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.modern_bert.modern_bert_backbone import (
    ModernBertBackbone,
)
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer
from keras_hub.src.utils.tensor_utils import tf


@keras_hub_export(
    [
        "keras_hub.tokenizers.ModernBertTokenizer",
        "keras_hub.models.ModernBertTokenizer",
    ]
)
class ModernBertTokenizer(BytePairTokenizer):
    """ModernBERT byte-level BPE tokenizer.

    This tokenizer configures the special token defaults required for the
    ModernBERT architecture, mapping padding, mask, and sequence boundary
    tokens to their specific unsplittable representations.

    Whitespace immediately preceding `[MASK]` is absorbed into the token, to
    match HuggingFace, which declares it as `AddedToken(..., lstrip=True)`.

    Args:
        vocabulary: dict or string. A dictionary mapping string tokens
        to integer IDs, or a file path to a json-serialized vocabulary map.

        merges: list or string. A list of byte pair merge rule strings,
        or a file path to a text merge rule list. Defaults to `None`.
        **kwargs: Additional keyword arguments passed to the parent
            `BytePairTokenizer` class.

    Examples:
    ```python
    import keras_hub

    # Load tokenizer directly from a preset configuration
    tokenizer = keras_hub.models.ModernBertTokenizer.from_preset(
        "modernbert_base_en"
    )

    # Encode raw text strings to integer ID tokens
    token_ids = tokenizer("The quick brown fox.")
    ```
    """

    backbone_cls = ModernBertBackbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        **kwargs,
    ):
        pad_token = "[PAD]"
        mask_token = "[MASK]"
        cls_token = "[CLS]"
        sep_token = "[SEP]"

        # Registered before `super().__init__()` so that
        # `_update_special_token_ids` sees them and raises on a vocabulary
        # that is missing one, rather than silently leaving the id `None`.
        # ModernBERT uses the EOS token for CLS and SEP.
        self._add_special_token(pad_token, "pad_token")
        self._add_special_token(mask_token, "mask_token")
        self._add_special_token(cls_token, "cls_token")
        self._add_special_token(sep_token, "sep_token")

        unsplittable_tokens = list(kwargs.pop("unsplittable_tokens", []))

        for token in (pad_token, mask_token, cls_token, sep_token):
            if token not in unsplittable_tokens:
                unsplittable_tokens.append(token)

        kwargs["unsplittable_tokens"] = unsplittable_tokens
        kwargs.setdefault(
            "add_prefix_space",
            False,
        )

        # Derived from `mask_token` so the two cannot drift. The pattern is
        # valid in both RE2 (`tf.strings.regex_replace`) and Python `re`.
        self._mask_lstrip_pattern = r"\s+" + re.escape(mask_token)

        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )

    def tokenize(self, inputs):
        inputs = self._lstrip_mask_token(inputs)
        return super().tokenize(inputs)

    def _lstrip_mask_token(self, inputs):
        """Absorb whitespace immediately preceding `[MASK]`.

        HF declares `[MASK]` as `AddedToken(..., lstrip=True)`, so any
        whitespace to the left of the token is consumed as part of the match.
        `unsplittable_tokens` is a flat list of strings with no channel for
        per-token `lstrip`/`rstrip` flags, so without this the preceding space
        survives the split and byte-level BPE emits it as a separate `Ġ`
        token, shifting every id after the mask.

        Normalizing here rather than in either backend keeps the TF graph path
        and the `tokenizers` path in agreement.
        """
        if tf is not None and isinstance(inputs, (tf.Tensor, tf.RaggedTensor)):
            # RE2 syntax, and graph-safe.
            return tf.strings.regex_replace(
                inputs, self._mask_lstrip_pattern, self.mask_token
            )
        if isinstance(inputs, bytes):
            inputs = inputs.decode("utf-8")
        if isinstance(inputs, str):
            return re.sub(self._mask_lstrip_pattern, self.mask_token, inputs)
        if isinstance(inputs, (list, tuple)):
            return [self._lstrip_mask_token(x) for x in inputs]
        # Anything else (e.g. already-tokenized ids) is passed through
        # untouched for the parent to validate.
        return inputs

    @property
    def start_token_id(self):
        return self.cls_token_id

    @property
    def end_token_id(self):
        return self.sep_token_id
