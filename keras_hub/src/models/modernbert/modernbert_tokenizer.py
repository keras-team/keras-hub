import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.modernbert.modernbert_backbone import (
    ModernBertBackbone,
)
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras.utils.register_keras_serializable(
    package="keras_hub",
)
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
        "modernbert_base"
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
        self.pad_token = "<|padding|>"
        self.mask_token = "[MASK]"
        self.start_token = "<|endoftext|>"
        self.end_token = "<|endoftext|>"

        unsplittable_tokens = kwargs.pop(
            "unsplittable_tokens",
            [],
        )

        unsplittable_tokens = list(unsplittable_tokens)

        special_tokens = [
            self.pad_token,
            self.mask_token,
            self.start_token,
            self.end_token,
        ]
        for token in special_tokens:
            if token not in unsplittable_tokens:
                unsplittable_tokens.append(token)

        kwargs["unsplittable_tokens"] = unsplittable_tokens

        kwargs["add_prefix_space"] = kwargs.get(
            "add_prefix_space",
            False,
        )

        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )

        self.cls_token = self.start_token
        self.sep_token = self.end_token

    @property
    def pad_token_id(self):
        return self._safe_token_id(self.pad_token)

    @property
    def mask_token_id(self):
        return self._safe_token_id(self.mask_token)

    @property
    def start_token_id(self):
        return self._safe_token_id(self.start_token)

    @property
    def end_token_id(self):
        return self._safe_token_id(self.end_token)

    def _safe_token_id(self, token):
        if self.vocabulary is None:
            return 0

        return self.token_to_id(token)

    @property
    def vocabulary_size(self):
        if self.vocabulary is None:
            return 0

        return len(self.vocabulary)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "vocabulary": self.vocabulary,
                "merges": self.merges,
            }
        )

        return config
