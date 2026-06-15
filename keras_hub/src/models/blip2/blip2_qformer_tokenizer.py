from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.blip2.blip2_backbone import BLIP2Backbone
from keras_hub.src.tokenizers.word_piece_tokenizer import WordPieceTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.BLIP2QFormerTokenizer",
        "keras_hub.models.BLIP2QFormerTokenizer",
    ]
)
class BLIP2QFormerTokenizer(WordPieceTokenizer):
    """Instruction tokenizer for the InstructBLIP Q-Former (WordPiece).

    InstructBLIP feeds the instruction text into the (instruction-aware)
    Q-Former using a BERT (`bert-base-uncased`) WordPiece tokenizer, separately
    from the LLaMA tokenizer used for the language-model prompt. This wrapper
    mirrors `keras_hub.models.BertTokenizer` but associates the tokenizer with
    `BLIP2Backbone` for `from_preset()` support.

    Args:
        vocabulary: A list of strings or a string filename path. If passing a
            list, each element should be a single WordPiece token string. If
            passing a filename, the file should contain one token per line.
        lowercase: bool. If `True`, input text is lowercased before
            tokenization. Defaults to `True` (matching `bert-base-uncased`).
        special_tokens_in_strings: bool. Whether the tokenizer should expect
            special tokens in input strings. Defaults to `False`.
    """

    backbone_cls = BLIP2Backbone

    def __init__(
        self,
        vocabulary=None,
        lowercase=True,
        **kwargs,
    ):
        # Use a dedicated preset config file / asset subdir so the Q-Former
        # WordPiece vocabulary never collides with the language-model
        # tokenizer's assets within a single InstructBLIP preset.
        kwargs.setdefault("config_file", "qformer_tokenizer.json")
        self._add_special_token("[CLS]", "cls_token")
        self._add_special_token("[SEP]", "sep_token")
        self._add_special_token("[PAD]", "pad_token")
        self._add_special_token("[MASK]", "mask_token")
        # Aliases for cross-tokenizer compatibility.
        self._add_special_token("[CLS]", "start_token")
        self._add_special_token("[SEP]", "end_token")
        super().__init__(
            vocabulary=vocabulary,
            lowercase=lowercase,
            **kwargs,
        )
