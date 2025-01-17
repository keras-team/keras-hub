from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.xlm_roberta.xlm_roberta_backbone import (
    XLMRobertaBackbone,
)
from keras_hub.src.tokenizers.sentence_piece_tokenizer import (
    SentencePieceTokenizer,
)
from keras_hub.src.utils.tensor_utils import tensor_to_list

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export(
    [
        "keras_hub.tokenizers.XLMRobertaTokenizer",
        "keras_hub.models.XLMRobertaTokenizer",
    ]
)
class XLMRobertaTokenizer(SentencePieceTokenizer):
    """An XLM-RoBERTa tokenizer using SentencePiece subword segmentation.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_hub.tokenizers.SentencePieceTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by
    XLM-RoBERTa models and provides a `from_preset()` method to automatically
    download a matching vocabulary for an XLM-RoBERTa preset.

    Note: If you are providing your own custom SentencePiece model, the original
    fairseq implementation of XLM-RoBERTa re-maps some token indices from the
    underlying sentencepiece output. To preserve compatibility, we do the same
    re-mapping here.

    If input is a batch of strings (rank > 0), the layer will output a
    `tf.RaggedTensor` where the last dimension of the output is ragged.

    If input is a scalar string (rank == 0), the layer will output a dense
    `tf.Tensor` with static shape `[None]`.

    Args:
        proto: Either a `string` path to a SentencePiece proto file or a
            `bytes` object with a serialized SentencePiece proto. See the
            [SentencePiece repository](https://github.com/google/sentencepiece)
            for more details on the format.

    Examples:
    ```python
    tokenizer = keras_hub.models.XLMRobertaTokenizer.from_preset(
        "xlm_roberta_base_multi",
    )

    # Unbatched inputs.
    tokenizer("the quick brown fox")

    # Batched inputs.
    tokenizer(["the quick brown fox", "الأرض كروية"])

    # Detokenization.
    tokenizer.detokenize(tokenizer("the quick brown fox"))

    # Custom vocabulary
    def train_sentencepiece(ds, vocab_size):
        bytes_io = io.BytesIO()
        sentencepiece.SentencePieceTrainer.train(
            sentence_iterator=ds.as_numpy_iterator(),
            model_writer=bytes_io,
            vocab_size=vocab_size,
            model_type="WORD",
            unk_id=0,
            bos_id=1,
            eos_id=2,
        )
        return bytes_io.getvalue()

    ds = tf.data.Dataset.from_tensor_slices(
        ["the quick brown fox", "the earth is round"]
    )
    proto = train_sentencepiece(ds, vocab_size=10)
    tokenizer = keras_hub.models.XLMRobertaTokenizer(proto=proto)
    ```
    """

    backbone_cls = XLMRobertaBackbone

    def __init__(self, proto, **kwargs):
        # Handle special tokens manually, as the tokenizer maps these tokens in
        # a way that is not reflected in the vocabulary.
        self.start_token, self.start_token_id = "<s>", 0
        self.pad_token, self.pad_token_id = "<pad>", 1
        self.end_token, self.end_token_id = "</s>", 2
        self.unk_token, self.unk_token_id = "<unk>", 3
        super().__init__(proto=proto, **kwargs)

    @property
    def special_tokens(self):
        return ["<s>", "<pad>", "</s>", "<unk>"]

    @property
    def special_token_ids(self):
        return [0, 1, 2, 3]

    def set_proto(self, proto):
        super().set_proto(proto)
        if proto is not None:
            self.mask_token_id = self.vocabulary_size() - 1
        else:
            self.mask_token_id = None

    def vocabulary_size(self):
        """Get the size of the tokenizer vocabulary."""
        return super().vocabulary_size() + 2

    def get_vocabulary(self):
        """Get the size of the tokenizer vocabulary."""
        self._check_vocabulary()
        vocabulary = tensor_to_list(
            self._sentence_piece.id_to_string(
                tf.range(super().vocabulary_size())
            )
        )
        return self._vocabulary_prefix + vocabulary[3:] + ["<mask>"]

    def id_to_token(self, id):
        """Convert an integer id to a string token."""
        self._check_vocabulary()

        if id == self.mask_token_id:
            return "<mask>"

        if id < len(self._vocabulary_prefix) and id >= 0:
            return self._vocabulary_prefix[id]

        if id - 1 >= super().vocabulary_size() or id - 1 < 0:
            raise ValueError(
                f"`id` must be in range [0, {self.vocabulary_size() - 1}]. "
                f"Received: {id}"
            )

        return tensor_to_list(self._sentence_piece.id_to_string(id - 1))

    def token_to_id(self, token):
        """Convert a string token to an integer id."""
        self._check_vocabulary()

        if token in self._vocabulary_prefix:
            return self._vocabulary_prefix.index(token)

        spm_token_id = self._sentence_piece.string_to_id(token)

        # OOV token
        spm_unk_token_id = self._sentence_piece.string_to_id("<unk>")
        if spm_token_id == spm_unk_token_id:
            return self.unk_token_id

        return int(spm_token_id.numpy()) + 1

    def tokenize(self, inputs):
        self._check_vocabulary()
        tokens = super().tokenize(inputs)

        # Correct `unk_token_id` (0 -> 3). Note that we do not correct
        # `start_token_id` and `end_token_id`; they are dealt with in
        # `XLMRobertaTextClassifierPreprocessor`.
        tokens = tf.where(tf.equal(tokens, 0), self.unk_token_id - 1, tokens)

        # Shift the tokens IDs right by one.
        return tf.add(tokens, 1)

    def detokenize(self, inputs):
        self._check_vocabulary()
        tokens = tf.ragged.boolean_mask(
            inputs, tf.not_equal(inputs, self.mask_token_id)
        )

        # Shift the tokens IDs left by one.
        tokens = tf.subtract(tokens, 1)

        # Correct `unk_token_id`, `end_token_id`, `start_token_id`,
        # respectively.
        # Note: The `pad_token_id` is taken as 0 (`unk_token_id`) since the
        # proto does not contain `pad_token_id`. This mapping of the pad token
        # is done automatically by the above subtraction.
        tokens = tf.where(tf.equal(tokens, self.unk_token_id - 1), 0, tokens)
        tokens = tf.where(tf.equal(tokens, self.end_token_id - 1), 2, tokens)
        tokens = tf.where(tf.equal(tokens, self.start_token_id - 1), 1, tokens)

        # Note: Even though we map `"<s>" and `"</s>"` to the correct IDs,
        # the `detokenize` method will return empty strings for these tokens.
        # This is a vagary of the `sentencepiece` library.
        return super().detokenize(tokens)
