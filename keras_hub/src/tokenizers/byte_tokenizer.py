import unicodedata

import keras
import numpy as np

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.tokenizers import tokenizer
from keras_hub.src.utils.tensor_utils import convert_to_ragged_batch
from keras_hub.src.utils.tensor_utils import in_tf_function
from keras_hub.src.utils.tensor_utils import is_int_dtype
from keras_hub.src.utils.tensor_utils import preprocessing_function

try:
    import tensorflow as tf
except ImportError:
    tf = None
try:
    import tensorflow_text as tf_text
except ImportError:
    tf_text = None


def _decode_with_replacement(byte_seq, errors, replacement_char):
    byte_seq = bytes(byte_seq)

    # If using a custom replacement character
    if errors == "replace" and replacement_char != 65533:
        result = []
        start = 0
        while start < len(byte_seq):
            try:
                # Try to decode the remainder of the sequence
                decoded = byte_seq[start:].decode("utf-8", errors="strict")
                result.append(decoded)
                break
            except UnicodeDecodeError as e:
                # Decode the valid chunk before the error
                valid_part = byte_seq[start : start + e.start].decode(
                    "utf-8", errors="strict"
                )
                result.append(valid_part)
                # Append the custom replacement character
                result.append(chr(replacement_char))
                # Skip past the invalid bytes reported by the error
                start = start + e.end

        return "".join(result)

    # Standard behavior for all other cases
    try:
        return byte_seq.decode("utf-8", errors=errors)
    except UnicodeDecodeError as e:
        raise ValueError(f"Invalid byte sequence: {e}")


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

        _allow_python_workflow = kwargs.pop("_allow_python_workflow", True)
        super().__init__(
            dtype=dtype, _allow_python_workflow=_allow_python_workflow, **kwargs
        )

        self.lowercase = lowercase
        self.sequence_length = sequence_length
        self.normalization_form = normalization_form
        self.errors = errors
        self.replacement_char = replacement_char

        self._char_lst = [i.tobytes() for i in np.arange(256, dtype=np.uint8)]
        self._update_special_token_ids()

    def vocabulary_size(self):
        """Get the integer size of the tokenizer vocabulary."""
        return 256

    def get_vocabulary(self):
        vocab = {}
        for i in range(self.vocabulary_size()):
            vocab[chr(i)] = i
        return vocab

    def tokenize(self, inputs):
        if not self._allow_python_workflow or in_tf_function():
            return self._tokenize_tf(inputs)
        else:
            return self._tokenize_python(inputs)

    @preprocessing_function
    def _tokenize_tf(self, inputs):
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

    def _tokenize_python(self, inputs):
        def _canonicalize_tokenize_inputs(inputs):
            if isinstance(inputs, str):
                return [inputs], False
            elif isinstance(inputs, (tuple, list)):
                if not all(isinstance(i, str) for i in inputs):
                    raise ValueError(
                        "If a list or tuple is provided as input, all elements "
                        "must be strings. "
                        f"Received: {inputs}"
                    )
                return list(inputs), True
            elif tf is not None and isinstance(inputs, tf.Tensor):
                unbatched = inputs.shape.rank == 0
                if unbatched:
                    inputs = tf.expand_dims(inputs, 0)
                inputs = inputs.numpy().tolist()
                inputs = keras.tree.map_structure(
                    lambda x: x.decode("utf-8"), inputs
                )
                return inputs, not unbatched
            else:
                raise ValueError(
                    "Input should be a string or a list of strings. "
                    f"Received: {inputs}"
                )

        inputs, batched = _canonicalize_tokenize_inputs(inputs)

        batched_tokens = []
        for text in inputs:
            if self.lowercase:
                text = text.casefold()
            if self.normalization_form is not None:
                text = unicodedata.normalize(self.normalization_form, text)
            # Convert to byte integers
            tokens = list(text.encode("utf-8"))
            batched_tokens.append(tokens)

        # Handle sequence_length truncation and padding
        if self.sequence_length:
            pad_token_id = getattr(self, "pad_token_id", 0)
            batched_tokens = [
                tokens[: self.sequence_length]
                + [pad_token_id] * max(0, self.sequence_length - len(tokens))
                for tokens in batched_tokens
            ]

        if not batched:
            batched_tokens = batched_tokens[0]
        return batched_tokens

    def detokenize(self, inputs):
        if not self._allow_python_workflow or in_tf_function():
            return self._detokenize_tf(inputs)
        else:
            return self._detokenize_python(inputs)

    @preprocessing_function
    def _detokenize_tf(self, inputs):
        inputs, unbatched, rectangular = convert_to_ragged_batch(inputs)
        # Remove trailing padding tokens, so that trailing "\x00" bytes don't
        # show up in the detokenized output.
        inputs = tf.ragged.boolean_mask(inputs, tf.not_equal(inputs, 0))

        _char_lst_tensor = tf.constant(self._char_lst)
        outputs = tf.strings.reduce_join(
            tf.gather(_char_lst_tensor, inputs), axis=-1
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

    def _detokenize_python(self, inputs):
        def _canonicalize_detokenize_inputs(inputs):
            if tf is not None and isinstance(
                inputs, (tf.Tensor, tf.RaggedTensor)
            ):
                if isinstance(inputs, tf.RaggedTensor):
                    inputs = inputs.to_list()
                else:
                    inputs = inputs.numpy().tolist()
            is_batched = True
            if isinstance(inputs, int):
                inputs = [[inputs]]
                is_batched = False
            elif isinstance(inputs, (tuple, list)):
                if not inputs or isinstance(inputs[0], int):
                    inputs = [list(inputs)]
                    is_batched = False
                else:
                    inputs = [list(seq) for seq in inputs]
            elif isinstance(inputs, np.ndarray) or keras.ops.is_tensor(inputs):
                inputs = keras.ops.convert_to_numpy(inputs)
                if inputs.ndim == 0:
                    inputs = [[inputs.item()]]
                    is_batched = False
                elif inputs.ndim == 1:
                    inputs = [inputs.tolist()]
                    is_batched = False
                elif inputs.ndim == 2:
                    inputs = inputs.tolist()
                else:
                    raise ValueError(
                        "Array must be 0, 1 or 2 dimensional. "
                        f"Received: {inputs.shape}"
                    )
            else:
                raise ValueError(
                    "Input should be an integer, a list of integers, backend "
                    f"tensor or numpy array. Received: {inputs}"
                )
            return inputs, is_batched

        inputs, batched = _canonicalize_detokenize_inputs(inputs)

        outputs = []
        for seq in inputs:
            # Remove padding tokens, so that trailing "\x00" bytes don't
            # show up in the detokenized output.
            # Using bytes().replace() executes directly in C for maximum speed
            seq_bytes = bytes(seq).replace(b"\x00", b"")

            decoded = _decode_with_replacement(
                seq_bytes, self.errors, self.replacement_char
            )
            outputs.append(decoded)

        if not batched:
            outputs = outputs[0]
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
