import numpy as np

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.tokenizers import tokenizer
from keras_hub.src.utils.tensor_utils import convert_to_ragged_batch
from keras_hub.src.utils.tensor_utils import is_int_dtype
from keras_hub.src.utils.tensor_utils import preprocessing_function

try:
    import tensorflow as tf
    import tensorflow_text as tf_text
except ImportError:
    tf = None
    tf_text = None


@keras_hub_export("keras_hub.tokenizers.ByteTokenizer")
class ByteTokenizer(tokenizer.Tokenizer):
    """Raw byte tokenizer.

    This tokenizer is a vocabulary-free tokenizer which will tokenize text as
    as raw bytes from [0, 256).

    Tokenizer outputs can either be padded and truncated with a
    `sequence_length` argument, or left un-truncated. The exact output will
    depend on the rank of the input tensors.

    If input is a batch of strings:
    By default, the layer will output a `tf.RaggedTensor` where the last
    dimension of the output is ragged. If `sequence_length` is set, the layer
    will output a dense `tf.Tensor` where all inputs have been padded or
    truncated to `sequence_length`.

    If input is a scalar string:
    There are two cases here. If `sequence_length` is set, the output will be
    a dense `tf.Tensor` of shape `[sequence_length]`. Otherwise, the output will
    be a dense `tf.Tensor` of shape `[None]`.

    The output dtype can be controlled via the
    `dtype` argument, which should be an integer type
    ("int16", "int32", etc.).

    Args:
        lowercase: boolean. If True, the input text will be converted to
            lowercase before tokenization.
        sequence_length: int. If set, the output will be converted to a dense
            tensor and padded/trimmed so all outputs are of sequence_length.
        normalization_form: string. One of the following values: (None, "NFC",
            "NFKC", "NFD", "NFKD"). If set, every UTF-8 string in the input
            tensor text will be normalized to the given form before tokenizing.
        errors: One of ('replace', 'remove', 'strict'). Specifies the
            `detokenize()` behavior when an invalid tokenizer is encountered.
            The value of `'strict'` will cause the operation to produce a
            `InvalidArgument` error on any invalid input formatting. A value of
            `'replace'` will cause the tokenizer to replace any invalid
            formatting in the input with the `replacement_char` codepoint.
            A value of `'ignore'` will cause the tokenizer to skip any invalid
            formatting in the input and produce no corresponding output
            character.
        replacement_char: int. The replacement character to
            use when an invalid byte sequence is encountered and when `errors`
            is set to "replace" (same behaviour as
            https://www.tensorflow.org/api_docs/python/tf/strings/unicode_transcode).
            (U+FFFD) is `65533`. Defaults to `65533`.

    Examples:

    Basic usage.
    >>> tokenizer = keras_hub.tokenizers.ByteTokenizer()
    >>> outputs = tokenizer("hello")
    >>> np.array(outputs)
    array([104, 101, 108, 108, 111], dtype=int32)

    Ragged outputs.
    >>> inputs = ["hello", "hi"]
    >>> tokenizer = keras_hub.tokenizers.ByteTokenizer()
    >>> seq1, seq2 = tokenizer(inputs)
    >>> np.array(seq1)
    array([104, 101, 108, 108, 111])
    >>> np.array(seq2)
    array([104, 105])

    Dense outputs.
    >>> inputs = ["hello", "hi"]
    >>> tokenizer = keras_hub.tokenizers.ByteTokenizer(sequence_length=8)
    >>> seq1, seq2 = tokenizer(inputs)
    >>> np.array(seq1)
    array([104, 101, 108, 108, 111,   0,   0,   0], dtype=int32)
    >>> np.array(seq2)
    array([104, 105,   0,   0,   0,   0,   0,   0], dtype=int32)

    Tokenize, then batch for ragged outputs.
    >>> tokenizer = keras_hub.tokenizers.ByteTokenizer()
    >>> ds = tf.data.Dataset.from_tensor_slices(["hello", "fun"])
    >>> ds = ds.map(tokenizer)
    >>> ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(2))
    >>> ds.take(1).get_single_element()
    <tf.RaggedTensor [[104, 101, 108, 108, 111], [102, 117, 110]]>

    Batch, then tokenize for ragged outputs.
    >>> tokenizer = keras_hub.tokenizers.ByteTokenizer()
    >>> ds = tf.data.Dataset.from_tensor_slices(["hello", "fun"])
    >>> ds = ds.batch(2).map(tokenizer)
    >>> ds.take(1).get_single_element()
    <tf.RaggedTensor [[104, 101, 108, 108, 111], [102, 117, 110]]>

    Tokenize, then batch for dense outputs (`sequence_length` provided).
    >>> tokenizer = keras_hub.tokenizers.ByteTokenizer(sequence_length=5)
    >>> ds = tf.data.Dataset.from_tensor_slices(["hello", "fun"])
    >>> ds = ds.map(tokenizer)
    >>> ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(2))
    >>> ds.take(1).get_single_element()
    <tf.Tensor: shape=(2, 5), dtype=int32, numpy=
    array([[104, 101, 108, 108, 111],
           [102, 117, 110,   0,   0]], dtype=int32)>

    Batch, then tokenize for dense outputs. (`sequence_length` provided).
    >>> tokenizer = keras_hub.tokenizers.ByteTokenizer(sequence_length=5)
    >>> ds = tf.data.Dataset.from_tensor_slices(["hello", "fun"])
    >>> ds = ds.batch(2).map(tokenizer)
    >>> ds.take(1).get_single_element()
    <tf.Tensor: shape=(2, 5), dtype=int32, numpy=
    array([[104, 101, 108, 108, 111],
           [102, 117, 110,   0,   0]], dtype=int32)>

    Detokenization.
    >>> inputs = [104, 101, 108, 108, 111]
    >>> tokenizer = keras_hub.tokenizers.ByteTokenizer()
    >>> tokenizer.detokenize(inputs)
    'hello'

    Detokenization with invalid bytes.
    >>> # The 255 below is invalid utf-8.
    >>> inputs = [104, 101, 255, 108, 108, 111]
    >>> tokenizer = keras_hub.tokenizers.ByteTokenizer(
    ...     errors="replace", replacement_char=88)
    >>> tokenizer.detokenize(inputs)
    'heXllo'
    """

    def __init__(
        self,
        lowercase=True,
        sequence_length=None,
        normalization_form=None,
        errors="replace",
        replacement_char=65533,
        dtype="int32",
        **kwargs,
    ):
        if not is_int_dtype(dtype):
            raise ValueError(
                f"Output dtype must be an integer type. Received: dtype={dtype}"
            )

        # Check normalization_form.
        if normalization_form not in (None, "NFC", "NFKC", "NFD", "NFKD"):
            raise ValueError(
                '`normalization_form` must be one of None, "NFC", "NFKC", '
                '"NFD", "NFKD". Received: normalization_form='
                f"{normalization_form}"
            )

        # Check errors.
        if errors not in ("strict", "replace", "ignore"):
            raise ValueError(
                '`errors` must be one of "strict", "replace", "ignore" '
                f"Received: errors={errors}"
            )

        super().__init__(dtype=dtype, **kwargs)

        self.lowercase = lowercase
        self.sequence_length = sequence_length
        self.normalization_form = normalization_form
        self.errors = errors
        self.replacement_char = replacement_char

        self._char_lst = tf.constant(
            [i.tobytes() for i in np.arange(256, dtype=np.uint8)]
        )
        self._update_special_token_ids()

    def vocabulary_size(self):
        """Get the integer size of the tokenizer vocabulary."""
        return 256

    def get_vocabulary(self):
        vocab = {}
        for i in range(self.vocabulary_size()):
            vocab[chr(i)] = i
        return vocab

    @preprocessing_function
    def tokenize(self, inputs):
        unbatched = inputs.shape.rank == 0
        if unbatched:
            inputs = tf.expand_dims(inputs, 0)

        # Optional: Lowercase the input.
        if self.lowercase:
            inputs = tf_text.case_fold_utf8(inputs)

        # Optional: Normalize unicode.
        if self.normalization_form is not None:
            inputs = tf_text.normalize_utf8(inputs, self.normalization_form)

        # Tokenize input strings.
        tokens = tf.strings.bytes_split(inputs)
        tokens = tf.squeeze(
            tf.ragged.map_flat_values(tf.io.decode_raw, tokens, tf.uint8), -1
        )
        tokens = tf.cast(tokens, self.compute_dtype)

        # Convert to a dense output if `sequence_length` is set.
        if self.sequence_length:
            output_shape = tokens.shape.as_list()
            output_shape[-1] = self.sequence_length
            tokens = tokens.to_tensor(shape=output_shape)

        if unbatched:
            tokens = tf.squeeze(tokens, 0)
        return tokens

    @preprocessing_function
    def detokenize(self, inputs):
        inputs, unbatched, rectangular = convert_to_ragged_batch(inputs)
        # Remove trailing padding tokens, so that trailing "\x00" bytes don't
        # show up in the detokenized output.
        inputs = tf.ragged.boolean_mask(inputs, tf.not_equal(inputs, 0))

        outputs = tf.strings.reduce_join(
            tf.gather(self._char_lst, inputs), axis=-1
        )

        # Handle errors if an invalid byte sequence is encountered.
        outputs = tf.strings.unicode_transcode(
            outputs,
            "UTF-8",
            "UTF-8",
            errors=self.errors,
            replacement_char=self.replacement_char,
        )
        if unbatched:
            outputs = tf.squeeze(outputs, 0)
        return outputs

    def id_to_token(self, id):
        """Convert an integer id to a string token."""
        if id >= self.vocabulary_size() or id < 0:
            raise ValueError(
                f"`id` must be in range [0, {self.vocabulary_size() - 1}]. "
                f"Received: {id}"
            )
        return chr(id)

    def token_to_id(self, token):
        """Convert a string token to an integer id."""
        id = ord(token)
        if id >= self.vocabulary_size():
            raise ValueError(
                f"Token {token} is not supported by `ByteTokenizer`."
            )
        return id

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lowercase": self.lowercase,
                "sequence_length": self.sequence_length,
                "normalization_form": self.normalization_form,
                "errors": self.errors,
                "replacement_char": self.replacement_char,
            }
        )
        return config
