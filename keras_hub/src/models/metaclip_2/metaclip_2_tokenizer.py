"""MetaCLIP 2 tokenizer implementation using XLM-V (SentencePiece)."""

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.metaclip_2.metaclip_2_backbone import (
    MetaCLIP2Backbone,
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
        "keras_hub.tokenizers.MetaCLIP2Tokenizer",
        "keras_hub.models.MetaCLIP2Tokenizer",
    ]
)
class MetaCLIP2Tokenizer(SentencePieceTokenizer):
    """MetaCLIP 2 tokenizer using SentencePiece subword segmentation.

    This tokenizer class tokenizes raw strings into integer sequences and
    is based on `keras_hub.tokenizers.SentencePieceTokenizer`. MetaCLIP 2 uses
    the XLM-V tokenizer (`facebook/xlm-v-base`) which is a multilingual
    SentencePiece BPE tokenizer with ~901K vocabulary supporting 100+ languages.

    Unlike the underlying tokenizer, it will check for all special tokens needed
    by MetaCLIP 2 models and provides a `from_preset()` method to automatically
    download a matching vocabulary for a MetaCLIP 2 preset.

    Note: The XLM-V tokenizer uses a remapping of special token indices similar
    to XLM-RoBERTa. The special tokens are mapped as follows:
    - `<s>` (BOS): 0
    - `<pad>`: 1
    - `</s>` (EOS): 2
    - `<unk>`: 3

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
    # Unbatched input.
    tokenizer = keras_hub.models.MetaCLIP2Tokenizer.from_preset(
        "metaclip_2_vit_huge_patch14_224"
    )
    tokenizer("The quick brown fox jumped.")

    # Batched input.
    tokenizer(["The quick brown fox jumped.", "The fox slept."])

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))

    # Multilingual support (XLM-V supports 100+ languages)
    tokenizer("这是一个测试")  # Chinese
    tokenizer("これはテストです")  # Japanese
    ```
    """

    backbone_cls = MetaCLIP2Backbone

    # Prefix tokens that are handled specially in XLM-V/XLM-RoBERTa
    _vocabulary_prefix = ["<s>", "<pad>", "</s>", "<unk>"]

    def __init__(self, proto, **kwargs):
        # Handle special tokens manually, as the tokenizer maps these tokens in
        # a way that is not reflected in the vocabulary (similar to XLM-RoBERTa). # noqa
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

    def vocabulary_size(self):
        """Get the size of the tokenizer vocabulary.

        The vocabulary size includes the original SentencePiece vocabulary
        plus additional special tokens.
        """
        return super().vocabulary_size() + 1

    def get_vocabulary(self):
        """Get the tokenizer vocabulary as a list of strings."""
        self._check_vocabulary()
        vocabulary = tensor_to_list(
            self._sentence_piece.id_to_string(
                tf.range(super().vocabulary_size())
            )
        )
        # Remap vocabulary: prefix tokens + rest of vocabulary (skip first 3)
        return self._vocabulary_prefix + vocabulary[3:]

    def id_to_token(self, id):
        """Convert an integer id to a string token."""
        self._check_vocabulary()

        if id < len(self._vocabulary_prefix) and id >= 0:
            return self._vocabulary_prefix[id]

        if id >= self.vocabulary_size() or id < 0:
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
        # the preprocessor.
        tokens = tf.where(tf.equal(tokens, 0), self.unk_token_id - 1, tokens)

        # Shift the tokens IDs right by one to account for vocabulary remapping.
        return tf.add(tokens, 1)

    def detokenize(self, inputs):
        self._check_vocabulary()

        # Shift the tokens IDs left by one.
        tokens = tf.subtract(inputs, 1)

        # Correct `unk_token_id`, `end_token_id`, `start_token_id`,
        # respectively.
        # Note: The `pad_token_id` is taken as 0 (`unk_token_id`) since the
        # proto does not contain `pad_token_id`. This mapping of the pad token
        # is done automatically by the above subtraction.
        tokens = tf.where(tf.equal(tokens, self.unk_token_id - 1), 0, tokens)
        tokens = tf.where(tf.equal(tokens, self.end_token_id - 1), 2, tokens)
        tokens = tf.where(tf.equal(tokens, self.start_token_id - 1), 1, tokens)

        # Filter out special tokens for cleaner output
        tokens = tf.ragged.boolean_mask(
            tokens,
            tf.logical_and(
                tf.not_equal(tokens, self.pad_token_id - 1),
                tf.logical_and(
                    tf.not_equal(tokens, self.start_token_id - 1),
                    tf.not_equal(tokens, self.end_token_id - 1),
                ),
            ),
        )

        return super().detokenize(tokens)
