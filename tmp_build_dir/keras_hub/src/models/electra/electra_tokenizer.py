from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.electra.electra_backbone import ElectraBackbone
from keras_hub.src.tokenizers.word_piece_tokenizer import WordPieceTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.ElectraTokenizer",
        "keras_hub.models.ElectraTokenizer",
    ]
)
class ElectraTokenizer(WordPieceTokenizer):
    """A ELECTRA tokenizer using WordPiece subword segmentation.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_hub.tokenizers.WordPieceTokenizer`.

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
    # Custom Vocabulary.
    vocab = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    vocab += ["The", "quick", "brown", "fox", "jumped", "."]

    # Instantiate the tokenizer.
    tokenizer = keras_hub.models.ElectraTokenizer(vocabulary=vocab)

    # Unbatched input.
    tokenizer("The quick brown fox jumped.")

    # Batched input.
    tokenizer(["The quick brown fox jumped.", "The fox slept."])

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))
    ```
    """

    backbone_cls = ElectraBackbone

    def __init__(
        self,
        vocabulary,
        lowercase=False,
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
            **kwargs,
        )
