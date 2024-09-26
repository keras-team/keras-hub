from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.deberta_v3.deberta_v3_backbone import (
    DebertaV3Backbone,
)
from keras_hub.src.tokenizers.sentence_piece_tokenizer import (
    SentencePieceTokenizer,
)

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export(
    [
        "keras_hub.tokenizers.DebertaV3Tokenizer",
        "keras_hub.models.DebertaV3Tokenizer",
    ]
)
class DebertaV3Tokenizer(SentencePieceTokenizer):
    """DeBERTa tokenizer layer based on SentencePiece.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_hub.tokenizers.SentencePieceTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by
    DeBERTa models and provides a `from_preset()` method to automatically
    download a matching vocabulary for a DeBERTa preset.

    If input is a batch of strings (rank > 0), the layer will output a
    `tf.RaggedTensor` where the last dimension of the output is ragged.

    If input is a scalar string (rank == 0), the layer will output a dense
    `tf.Tensor` with static shape `[None]`.

    Note: The mask token (`"[MASK]"`) is handled differently in this tokenizer.
    If the token is not present in the provided SentencePiece vocabulary, the
    token will be appended to the vocabulary. For example, if the vocabulary
    size is 100, the mask token will be assigned the ID 100.

    Args:
        proto: Either a `string` path to a SentencePiece proto file, or a
            `bytes` object with a serialized SentencePiece proto. See the
            [SentencePiece repository](https://github.com/google/sentencepiece)
            for more details on the format.

    Examples:

    ```python
    # Unbatched input.
    tokenizer = keras_hub.models.DebertaV3Tokenizer.from_preset(
        "deberta_v3_base_en",
    )
    tokenizer("The quick brown fox jumped.")

    # Batched inputs.
    tokenizer(["the quick brown fox", "the earth is round"])

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))

    # Custom vocabulary.
    bytes_io = io.BytesIO()
    ds = tf.data.Dataset.from_tensor_slices(["The quick brown fox jumped."])
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=ds.as_numpy_iterator(),
        model_writer=bytes_io,
        vocab_size=9,
        model_type="WORD",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_piece="[PAD]",
        bos_piece="[CLS]",
        eos_piece="[SEP]",
        unk_piece="[UNK]",
    )
    tokenizer = keras_hub.models.DebertaV3Tokenizer(
        proto=bytes_io.getvalue(),
    )
    tokenizer("The quick brown fox jumped.")
    ```
    """

    backbone_cls = DebertaV3Backbone

    def __init__(self, proto, **kwargs):
        self._add_special_token("[CLS]", "cls_token")
        self._add_special_token("[SEP]", "sep_token")
        self._add_special_token("[PAD]", "pad_token")
        # Also add `tokenizer.start_token` and `tokenizer.end_token` for
        # compatibility with other tokenizers.
        self._add_special_token("[CLS]", "start_token")
        self._add_special_token("[SEP]", "end_token")
        # Handle mask separately as it's not always in the vocab.
        self.mask_token = "[MASK]"
        self.mask_token_id = None
        super().__init__(proto=proto, **kwargs)

    @property
    def special_tokens(self):
        return super().special_tokens + [self.mask_token]

    @property
    def special_token_ids(self):
        return super().special_token_ids + [self.mask_token_id]

    def set_proto(self, proto):
        super().set_proto(proto)
        if proto is not None:
            if self.mask_token in super().get_vocabulary():
                self.mask_token_id = super().token_to_id(self.mask_token)
            else:
                self.mask_token_id = super().vocabulary_size()
        else:
            self.mask_token_id = None

    def vocabulary_size(self):
        sentence_piece_size = super().vocabulary_size()
        if sentence_piece_size == self.mask_token_id:
            return sentence_piece_size + 1
        return sentence_piece_size

    def get_vocabulary(self):
        sentence_piece_vocabulary = super().get_vocabulary()
        if self.mask_token_id is None:
            return sentence_piece_vocabulary
        if self.mask_token_id < super().vocabulary_size():
            return sentence_piece_vocabulary
        return sentence_piece_vocabulary + ["[MASK]"]

    def id_to_token(self, id):
        if id == self.mask_token_id:
            return "[MASK]"
        return super().id_to_token(id)

    def token_to_id(self, token):
        if token == "[MASK]":
            return self.mask_token_id
        return super().token_to_id(token)

    def detokenize(self, ids):
        ids = tf.ragged.boolean_mask(ids, tf.not_equal(ids, self.mask_token_id))
        return super().detokenize(ids)
