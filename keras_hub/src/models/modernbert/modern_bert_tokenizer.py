import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.modernbert.modern_bert_backbone import (
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
        pad_token = "<|padding|>"
        mask_token = "[MASK]"
        cls_token = "<|endoftext|>"
        sep_token = "<|endoftext|>"

        unsplittable_tokens = list(kwargs.pop("unsplittable_tokens", []))

        for token in (
            pad_token,
            mask_token,
            cls_token,
            sep_token,
        ):
            if token not in unsplittable_tokens:
                unsplittable_tokens.append(token)

        kwargs["unsplittable_tokens"] = unsplittable_tokens
        kwargs.setdefault(
            "add_prefix_space",
            False,
        )

        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )

        # Register special tokens using KerasHub pattern.
        self._add_special_token(
            pad_token,
            "pad_token",
        )

        self._add_special_token(
            mask_token,
            "mask_token",
        )

        # ModernBERT uses EOS token for CLS and SEP.
        self._add_special_token(
            cls_token,
            "cls_token",
        )

        self._add_special_token(
            sep_token,
            "sep_token",
        )

    @property
    def start_token_id(self):
        return self.cls_token_id

    @property
    def end_token_id(self):
        return self.sep_token_id

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "vocabulary": self.vocabulary,
                "merges": self.merges,
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        config = config.copy()

        dtype = config.get("dtype")
        if isinstance(dtype, dict):
            dtype_config = dtype.get("config", {})
            if isinstance(dtype_config, dict):
                config["dtype"] = dtype_config.get("name", "int32")

        vocabulary = config.pop("vocabulary", None)
        merges = config.pop("merges", None)

        return cls(
            vocabulary=vocabulary,
            merges=merges,
            **config,
        )
