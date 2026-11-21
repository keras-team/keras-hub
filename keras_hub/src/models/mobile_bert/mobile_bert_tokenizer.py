from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.mobile_bert.mobile_bert_backbone import (
    MobileBertBackbone,
)
from keras_hub.src.tokenizers.word_piece_tokenizer import WordPieceTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.MobileBertTokenizer",
        "keras_hub.models.MobileBertTokenizer",
    ]
)
class MobileBertTokenizer(WordPieceTokenizer):
    """A MobileBERT tokenizer using WordPiece subword segmentation.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_hub.tokenizers.WordPieceTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by
    MobileBERT models and provides a `from_preset()` method to automatically
    download a matching vocabulary for a MobileBERT preset.

    Args:
        vocabulary: list or str. A list of strings or a string filename path.
            If passing a list, each element of the list should be a single word
            piece token string. If passing a filename, the file should be a
            plain text file containing a single word piece token per line.
            Defaults to `None`.
        lowercase: bool. If `True`, the input text will be first lowered before
            tokenization. Defaults to `False`.
        special_tokens_in_strings: bool. A bool to indicate if the tokenizer
            should expect special tokens in input strings that should be
            tokenized and mapped correctly to their ids. Defaults to `False`.

    Examples:
    ```python
    # Unbatched input.
    tokenizer = keras_hub.models.MobileBertTokenizer.from_preset(
        "mobilebert_uncased",
    )
    tokenizer("The quick brown fox jumped.")

    # Batched input.
    tokenizer(["The quick brown fox jumped.", "The fox slept."])

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))

    # Custom vocabulary.
    vocab = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    vocab += ["The", "quick", "brown", "fox", "jumped", "."]
    tokenizer = keras_hub.models.MobileBertTokenizer(vocabulary=vocab)
    tokenizer("The quick brown fox jumped.")
    ```
    """

    backbone_cls = MobileBertBackbone

    def __init__(
        self,
        vocabulary=None,
        lowercase=False,
        special_tokens_in_strings=False,
        **kwargs,
    ):
        self._add_special_token("[CLS]", "cls_token")
        self._add_special_token("[SEP]", "sep_token")
        self._add_special_token("[PAD]", "pad_token")
        self._add_special_token("[MASK]", "mask_token")
        # Also add `tokenizer.start_token` and `tokenizer.end_token` for
        # compatibility with other tokenizers.
        self._add_special_token("[CLS]", "start_token")
        self._add_special_token("[SEP]", "end_token")
        super().__init__(
            vocabulary=vocabulary,
            lowercase=lowercase,
            special_tokens_in_strings=special_tokens_in_strings,
            **kwargs,
        )
