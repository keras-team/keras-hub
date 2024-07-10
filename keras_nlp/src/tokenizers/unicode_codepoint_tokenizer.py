# Copyright 2023 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.tokenizers import tokenizer
from keras_nlp.src.utils.tensor_utils import convert_to_ragged_batch
from keras_nlp.src.utils.tensor_utils import is_int_dtype

try:
    import tensorflow as tf
    import tensorflow_text as tf_text
except ImportError:
    tf = None
    tf_text = None


@keras_nlp_export("keras_nlp.tokenizers.UnicodeCodepointTokenizer")
class UnicodeCodepointTokenizer(tokenizer.Tokenizer):
    """A unicode character tokenizer layer.

    This tokenizer is a vocabulary free tokenizer which tokenizes text as
    unicode character codepoints.

    Tokenizer outputs can either be padded and truncated with a
    `sequence_length` argument, or left un-truncated. The exact output will
    depend on the rank of the input tensors.

    If input is a batch of strings (rank > 0):
    By default, the layer will output a `tf.RaggedTensor` where the last
    dimension of the output is ragged. If `sequence_length` is set, the layer
    will output a dense `tf.Tensor` where all inputs have been padded or
    truncated to `sequence_length`.

    If input is a scalar string (rank == 0):
    By default, the layer will output a dense `tf.Tensor` with static shape
    `[None]`. If `sequence_length` is set, the output will be
    a dense `tf.Tensor` of shape `[sequence_length]`.

    The output dtype can be controlled via the `dtype` argument, which should be
    an integer type ("int16", "int32", etc.).

    Args:
        lowercase: If `True`, the input text will be first lowered before
            tokenization.
        sequence_length: If set, the output will be converted to a dense
            tensor and padded/trimmed so all outputs are of sequence_length.
        normalization_form: One of the following string values (None, 'NFC',
            'NFKC', 'NFD', 'NFKD'). If set will normalize unicode to the given
            form before tokenizing.
        errors: One of ('replace', 'remove', 'strict'). Specifies the
            `detokenize()` behavior when an invalid codepoint is encountered.
            The value of `'strict'` will cause the tokenizer to produce a
            `InvalidArgument` error on any invalid input formatting. A value of
            `'replace'` will cause the tokenizer to replace any invalid
            formatting in the input with the replacement_char codepoint.
            A value of `'ignore'` will cause the tokenizer to skip any invalid
            formatting in the input and produce no corresponding output
            character.
        replacement_char: The unicode codepoint to use in place of invalid
            codepoints. (U+FFFD) is `65533`. Defaults to `65533`.
        input_encoding: One of ("UTF-8", "UTF-16-BE", or "UTF-32-BE").
            One of The encoding of the input text. Defaults to `"UTF-8"`.
        output_encoding: One of ("UTF-8", "UTF-16-BE", or "UTF-32-BE").
            The encoding of the output text. Defaults to `"UTF-8"`.
        vocabulary_size: Set the vocabulary `vocabulary_size`,
            by clamping all codepoints to the range [0, vocabulary_size).
            Effectively this will make the `vocabulary_size - 1` id the
            the OOV value.

    Examples:

    Basic Usage.
    >>> inputs = "Unicode Tokenizer"
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCodepointTokenizer()
    >>> outputs = tokenizer(inputs)
    >>> np.array(outputs)
    array([117, 110, 105,  99, 111, 100, 101,  32, 116, 111, 107, 101, 110,
        105, 122, 101, 114], dtype=int32)

    Ragged outputs.
    >>> inputs = ["पुस्तक", "کتاب"]
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCodepointTokenizer()
    >>> seq1, seq2 = tokenizer(inputs)
    >>> np.array(seq1)
    array([2346, 2369, 2360, 2381, 2340, 2325], dtype=int32)
    >>> np.array(seq2)
    array([1705, 1578, 1575, 1576], dtype=int32)

    Dense outputs.
    >>> inputs = ["पुस्तक", "کتاب"]
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCodepointTokenizer(
    ...     sequence_length=8)
    >>> seq1, seq2 = tokenizer(inputs)
    >>> np.array(seq1)
    array([2346, 2369, 2360, 2381, 2340, 2325,    0,    0], dtype=int32)
    >>> np.array(seq2)
    array([1705, 1578, 1575, 1576,    0,    0,    0,    0], dtype=int32)

    Tokenize, then batch for ragged outputs.
    >>> inputs = ["Book", "पुस्तक", "کتاب"]
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCodepointTokenizer()
    >>> ds = tf.data.Dataset.from_tensor_slices(inputs)
    >>> ds = ds.map(tokenizer)
    >>> ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(3))
    >>> ds.take(1).get_single_element()
    <tf.RaggedTensor [[98, 111, 111, 107],
        [2346, 2369, 2360, 2381, 2340, 2325],
        [1705, 1578, 1575, 1576]]>

    Batch, then tokenize for ragged outputs.
    >>> inputs = ["Book", "पुस्तक", "کتاب"]
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCodepointTokenizer()
    >>> ds = tf.data.Dataset.from_tensor_slices(inputs)
    >>> ds = ds.batch(3).map(tokenizer)
    >>> ds.take(1).get_single_element()
    <tf.RaggedTensor [[98, 111, 111, 107],
        [2346, 2369, 2360, 2381, 2340, 2325],
        [1705, 1578, 1575, 1576]]>

    Tokenize, then batch for dense outputs (`sequence_length` provided).
    >>> inputs = ["Book", "पुस्तक", "کتاب"]
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCodepointTokenizer(
    ...     sequence_length=5)
    >>> ds = tf.data.Dataset.from_tensor_slices(inputs)
    >>> ds = ds.map(tokenizer)
    >>> ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(3))
    >>> ds.take(1).get_single_element()
    <tf.Tensor: shape=(3, 5), dtype=int32, numpy=
    array([[  98,  111,  111,  107,    0],
        [2346, 2369, 2360, 2381, 2340],
        [1705, 1578, 1575, 1576,    0]], dtype=int32)>

    Batch, then tokenize for dense outputs (`sequence_length` provided).
    >>> inputs = ["Book", "पुस्तक", "کتاب"]
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCodepointTokenizer(
    ...     sequence_length=5)
    >>> ds = tf.data.Dataset.from_tensor_slices(inputs)
    >>> ds = ds.batch(3).map(tokenizer)
    >>> ds.take(1).get_single_element()
    <tf.Tensor: shape=(3, 5), dtype=int32, numpy=
    array([[  98,  111,  111,  107,    0],
        [2346, 2369, 2360, 2381, 2340],
        [1705, 1578, 1575, 1576,    0]], dtype=int32)>

    Tokenization with truncation.
    >>> inputs = ["I Like to Travel a Lot", "मैं किताबें पढ़ना पसंद करता हूं"]
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCodepointTokenizer(
    ...     sequence_length=5)
    >>> outputs = tokenizer(inputs)
    >>> np.array(outputs)
    array([[ 105,   32,  108,  105,  107],
           [2350, 2376, 2306,   32, 2325]], dtype=int32)

    Tokenization with vocabulary_size.
    >>> latin_ext_cutoff = 592
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCodepointTokenizer(
    ...     vocabulary_size=latin_ext_cutoff)
    >>> outputs = tokenizer("¿Cómo estás?")
    >>> np.array(outputs)
    array([191,  99, 243, 109, 111,  32, 101, 115, 116, 225, 115,  63],
          dtype=int32)
    >>> outputs = tokenizer("आप कैसे हैं")
    >>> np.array(outputs)
    array([591, 591,  32, 591, 591, 591, 591,  32, 591, 591, 591],
          dtype=int32)

    Detokenization.
    >>> inputs = tf.constant([110, 105, 110, 106,  97], dtype="int32")
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCodepointTokenizer()
    >>> outputs = tokenizer.detokenize(inputs)
    >>> np.array(outputs).astype("U")
    array('ninja', dtype='<U5')

    Detokenization with padding.
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCodepointTokenizer(
    ...     sequence_length=7)
    >>> dataset = tf.data.Dataset.from_tensor_slices(["a b c", "b c", "a"])
    >>> dataset = dataset.map(tokenizer)
    >>> dataset.take(1).get_single_element()
    <tf.Tensor: shape=(7,), dtype=int32,
        numpy=array([97, 32, 98, 32, 99,  0,  0], dtype=int32)>
    >>> detokunbatched = dataset.map(tokenizer.detokenize)
    >>> detokunbatched.take(1).get_single_element()
    <tf.Tensor: shape=(), dtype=string, numpy=b'a b c'>

    Detokenization with invalid bytes.
    >>> inputs = tf.constant([110, 105, 10000000, 110, 106,  97])
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCodepointTokenizer(
    ...     errors="replace", replacement_char=88)
    >>> outputs = tokenizer.detokenize(inputs)
    >>> np.array(outputs).astype("U")
    array('niXnja', dtype='<U6')
    """

    def __init__(
        self,
        sequence_length=None,
        lowercase=True,
        normalization_form=None,
        errors="replace",
        replacement_char=65533,
        input_encoding="UTF-8",
        output_encoding="UTF-8",
        vocabulary_size=None,
        dtype="int32",
        **kwargs,
    ) -> None:
        if not is_int_dtype(dtype):
            raise ValueError(
                "Output dtype must be an integer type. "
                f"Received: dtype={dtype}"
            )

        # Check normalization_form.
        if normalization_form not in [None, "NFC", "NFKC", "NFD", "NFKD"]:
            raise ValueError(
                '`normalization_form` must be one of None, "NFC", "NFKC", '
                '"NFD", "NFKD". Received: normalization_form='
                f"{normalization_form}"
            )

        # Check errors.
        if errors not in ["strict", "replace", "ignore"]:
            raise ValueError(
                '`errors` must be one of "strict", "replace", "ignore" '
                f"Received: errors={errors}"
            )

        # Check normalization_form matches input_encoding.
        if normalization_form:
            if input_encoding != "UTF-8":
                raise ValueError(
                    """Normalization Forms are Only Supported for Input Encoding
                     UTF-8"""
                )

        super().__init__(dtype=dtype, **kwargs)

        self.sequence_length = sequence_length
        self.lowercase = lowercase
        self.normalization_form = normalization_form
        self.errors = errors
        self.replacement_char = replacement_char
        self.input_encoding = input_encoding
        self.output_encoding = output_encoding
        self._vocabulary_size = vocabulary_size

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "lowercase": self.lowercase,
                "normalization_form": self.normalization_form,
                "errors": self.errors,
                "replacement_char": self.replacement_char,
                "input_encoding": self.input_encoding,
                "output_encoding": self.output_encoding,
                "vocabulary_size": self._vocabulary_size,
            }
        )
        return config

    def vocabulary_size(self):
        """Get the size of the tokenizer vocabulary. None implies no vocabulary
        size was provided"""
        return self._vocabulary_size

    def get_vocabulary(self):
        vocab = {}
        for i in range(self.vocabulary_size()):
            vocab[chr(i)] = i
        return vocab

    def tokenize(self, inputs):
        if not isinstance(inputs, (tf.Tensor, tf.RaggedTensor)):
            inputs = tf.convert_to_tensor(inputs)

        scalar_input = inputs.shape.rank == 0
        if scalar_input:
            inputs = tf.expand_dims(inputs, 0)

        # Optionally lowercase the text
        if self.lowercase:
            inputs = tf_text.case_fold_utf8(inputs)

        # Optionally normalize the text to a given form
        if self.normalization_form:
            inputs = tf_text.normalize_utf8(inputs, self.normalization_form)

        tokens = tf.strings.unicode_decode(
            inputs,
            errors=self.errors,
            replacement_char=self.replacement_char,
            input_encoding=self.input_encoding,
        )
        tokens = tf.cast(tokens, self.compute_dtype)

        if self.sequence_length:
            output_shape = tokens.shape.as_list()
            output_shape[-1] = self.sequence_length
            tokens = tokens.to_tensor(shape=output_shape)

        if scalar_input:
            tokens = tf.squeeze(tokens, 0)

        # Optionally clamps the output code point values to be in the
        # range [0, vocabulary_size)
        if self._vocabulary_size:
            tokens = tf.clip_by_value(tokens, 0, self._vocabulary_size - 1)

        return tokens

    def detokenize(self, inputs):
        inputs, unbatched, _ = convert_to_ragged_batch(inputs)
        inputs = tf.ragged.boolean_mask(inputs, tf.not_equal(inputs, 0))
        outputs = tf.strings.unicode_encode(
            inputs,
            errors=self.errors,
            replacement_char=self.replacement_char,
            output_encoding=self.output_encoding,
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
                f"Token {token} is not supported by `UnicodeCodepointTokenizer`."
            )
        return id
