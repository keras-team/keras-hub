from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.esm.esm_backbone import ESMBackbone
from keras_hub.src.tokenizers.word_piece_tokenizer import WordPieceTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.ESMTokenizer",
        "keras_hub.models.ESMTokenizer",
    ]
)
class ESMTokenizer(WordPieceTokenizer):
    """A ESM tokenizer using WordPiece subword segmentation.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_hub.tokenizers.WordPieceTokenizer`. Unlike the
    underlying tokenizer, it will check for special tokens needed by ESM
    models and provides a `from_preset()` method to automatically download
    a matching vocabulary for a ESM preset.

    If input is a batch of strings (rank > 0), the layer will output a
    `tf.RaggedTensor` where the last dimension of the output is ragged.

    If input is a scalar string (rank == 0), the layer will output a dense
    `tf.Tensor` with static shape `[None]`.

    Args:
        vocabulary: A list of strings or a string filename path. If
            passing a list, each element of the list should be a single word
            piece token string. If passing a filename, the file should be a
            plain text file containing a single word piece token per line.
        lowercase: If `True`, the input text will be first lowered before
            tokenization.
        special_tokens_in_strings: bool. A bool to indicate if the tokenizer
            should expect special tokens in input strings that should be
            tokenized and mapped correctly to their ids. Defaults to False.

    Examples:
    ```python
    # Unbatched input.
    tokenizer = keras_hub.models.ESMTokenizer.from_preset(
        "hf://facebook/esm2_t6_8M_UR50D",
    )
    tokenizer("The quick brown fox jumped.")

    # Batched input.
    tokenizer(["The quick brown fox jumped.", "The fox slept."])

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))

    # Custom vocabulary.
    vocab = ["[UNK]", "<cls>", "<eos>", "<pad>", "<mask>"]
    vocab += ["The", "quick", "brown", "fox", "jumped", "."]
    tokenizer = keras_hub.models.ESMTokenizer(vocabulary=vocab)
    tokenizer("The quick brown fox jumped.")
    ```
    """

    backbone_cls = ESMBackbone

    def __init__(
        self,
        vocabulary=None,
        lowercase=False,
        oov_token="<unk>",
        **kwargs,
    ):
        self._add_special_token("<cls>", "cls_token")
        self._add_special_token("<eos>", "sep_token")
        self._add_special_token("<pad>", "pad_token")
        self._add_special_token("<mask>", "mask_token")
        # Also add `tokenizer.start_token` and `tokenizer.end_token` for
        # compatibility with other tokenizers.
        self._add_special_token("<cls>", "start_token")
        self._add_special_token("<eos>", "end_token")
        super().__init__(
            vocabulary=vocabulary,
            lowercase=lowercase,
            oov_token=oov_token,
            **kwargs,
        )
