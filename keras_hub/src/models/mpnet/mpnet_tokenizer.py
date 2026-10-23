from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.mpnet.mpnet_backbone import MPNetBackbone
from keras_hub.src.tokenizers.word_piece_tokenizer import WordPieceTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.MPNetTokenizer",
        "keras_hub.models.MPNetTokenizer",
    ]
)
class MPNetTokenizer(WordPieceTokenizer):
    """An MPNet tokenizer using WordPiece subword segmentation.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_hub.tokenizers.WordPieceTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by MPNet
    models and provides a `from_preset()` method to automatically download
    a matching vocabulary for an MPNet preset.

    MPNet uses a BERT-style WordPiece vocabulary with RoBERTa-style special
    tokens: `"<s>"`, `"</s>"`, `"<pad>"` and `"<mask>"`. Out-of-vocabulary
    words are mapped to `"[UNK]"`.

    If input is a batch of strings (rank > 0), the layer will output a
    `tf.RaggedTensor` where the last dimension of the output is ragged.

    If input is a scalar string (rank == 0), the layer will output a dense
    `tf.Tensor` with static shape `[None]`.

    Args:
        vocabulary: A list of strings or a string filename path. If
            passing a list, each element of the list should be a single word
            piece token string. If passing a filename, the file should be a
            plain text file containing a single word piece token per line.
        lowercase: bool. If `True`, the input text will be lowercased before
            tokenization. Defaults to `True`.
        strip_accents: bool or `None`. If `True`, all accent marks will be
            removed from the text before tokenization. If `None`, accents are
            stripped if and only if `lowercase` is `True`, which matches the
            original MPNet tokenizer. Defaults to `None`.
        special_tokens_in_strings: bool. A bool to indicate if the tokenizer
            should expect special tokens in input strings that should be
            tokenized and mapped correctly to their ids. Defaults to False.

    Examples:
    ```python
    # Unbatched input.
    tokenizer = keras_hub.models.MPNetTokenizer.from_preset(
        "all_mpnet_base_v2_en",
    )
    tokenizer("The quick brown fox jumped.")

    # Batched input.
    tokenizer(["The quick brown fox jumped.", "The fox slept."])

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))

    # Custom vocabulary.
    vocab = ["<s>", "<pad>", "</s>", "[UNK]", "<mask>"]
    vocab += ["the", "quick", "brown", "fox", "jumped", "."]
    tokenizer = keras_hub.models.MPNetTokenizer(vocabulary=vocab)
    tokenizer("The quick brown fox jumped.")
    ```
    """

    backbone_cls = MPNetBackbone

    def __init__(
        self,
        vocabulary=None,
        lowercase=True,
        strip_accents=None,
        **kwargs,
    ):
        if strip_accents is None:
            strip_accents = lowercase
        self._add_special_token("<s>", "cls_token")
        self._add_special_token("</s>", "sep_token")
        self._add_special_token("<pad>", "pad_token")
        self._add_special_token("<mask>", "mask_token")
        # Also add `tokenizer.start_token` and `tokenizer.end_token` for
        # compatibility with other tokenizers.
        self._add_special_token("<s>", "start_token")
        self._add_special_token("</s>", "end_token")
        super().__init__(
            vocabulary=vocabulary,
            lowercase=lowercase,
            strip_accents=strip_accents,
            **kwargs,
        )
