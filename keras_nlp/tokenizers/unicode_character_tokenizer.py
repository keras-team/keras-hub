# Copyright 2022 The KerasNLP Authors
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

from typing import Any
from typing import Dict

import tensorflow as tf
import tensorflow_text as tf_text

from keras_nlp.tokenizers import tokenizer


class UnicodeCharacterTokenizer(tokenizer.Tokenizer):
    """A unicode character tokenizer layer.

    This tokenizer is a vocabulary free tokenizer which tokenizes text as
    unicode characters codepoints.

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
    an integer type (tf.int16, tf.int32, etc.).

    Args:
        lowercase: If true, the input text will be first lowered before
            tokenization.
        sequence_length: If set, the output will be converted to a dense
            tensor and padded/trimmed so all outputs are of sequence_length.
        normalization_form: One of the following string values (None, 'NFC',
            'NFKC', 'NFD', 'NFKD'). If set will normalize unicode to the given
            form before tokenizing.
        errors: One of ('replace', 'remove', 'strict'). Specifies the
            `detokenize()` behavior when an invalid codepoint is encountered.
            (same behavior as
            https://www.tensorflow.org/api_docs/python/tf/strings/unicode_transcode)
        replacement_char: The unicode codepoint to use in place of invalid
            codepoints. Defaults to 65533 (U+FFFD).
        input_encoding: One of ("UTF-8", "UTF-16-BE", or "UTF-32-BE").
            One of The encoding of the input text. Defaults to "UTF-8".
        output_encoding: One of ("UTF-8", "UTF-16-BE", or "UTF-32-BE").
            The encoding of the output text. Defaults to "UTF-8".
        vocabulary_size: Set the vocabulary `vocabulary_size`,
            by clamping all codepoints to the range [0, vocabulary_size).
            Effectively this will make the `vocabulary_size - 1` id the
            the OOV value.

    Examples:

    Basic Usage.
    >>> inputs = "Unicode Tokenizer"
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCharacterTokenizer()
    >>> tokenizer(inputs)
    <tf.Tensor: shape=(17,), dtype=int32, numpy=
    array([117, 110, 105,  99, 111, 100, 101,  32, 116, 111, 107, 101, 110,
        105, 122, 101, 114], dtype=int32)>

    Ragged outputs.
    >>> inputs = ["Book", "पुस्तक", "کتاب"]
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCharacterTokenizer()
    >>> tokenizer(inputs)
    <tf.RaggedTensor [[98, 111, 111, 107],
        [2346, 2369, 2360, 2381, 2340, 2325],
        [1705, 1578, 1575, 1576]]>

    Dense outputs.
    >>> inputs = ["Book", "पुस्तक", "کتاب"]
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCharacterTokenizer(
    ...     sequence_length=8)
    >>> tokenizer(inputs)
    <tf.Tensor: shape=(3, 8), dtype=int32, numpy=
    array([[  98,  111,  111,  107,    0,    0,    0,    0],
        [2346, 2369, 2360, 2381, 2340, 2325,    0,    0],
        [1705, 1578, 1575, 1576,    0,    0,    0,    0]], dtype=int32)>

    Tokenize, then batch for ragged outputs.
    >>> inputs = ["Book", "पुस्तक", "کتاب"]
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCharacterTokenizer()
    >>> ds = tf.data.Dataset.from_tensor_slices(inputs)
    >>> ds = ds.map(tokenizer)
    >>> ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(3))
    >>> ds.take(1).get_single_element()
    <tf.RaggedTensor [[98, 111, 111, 107],
        [2346, 2369, 2360, 2381, 2340, 2325],
        [1705, 1578, 1575, 1576]]>

    Batch, then tokenize for ragged outputs.
    >>> inputs = ["Book", "पुस्तक", "کتاب"]
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCharacterTokenizer()
    >>> ds = tf.data.Dataset.from_tensor_slices(inputs)
    >>> ds = ds.batch(3).map(tokenizer)
    >>> ds.take(1).get_single_element()
    <tf.RaggedTensor [[98, 111, 111, 107],
        [2346, 2369, 2360, 2381, 2340, 2325],
        [1705, 1578, 1575, 1576]]>

    Tokenize, then batch for dense outputs (`sequence_length` provided).
    >>> inputs = ["Book", "पुस्तक", "کتاب"]
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCharacterTokenizer(
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
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCharacterTokenizer(
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
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCharacterTokenizer(
    ...     sequence_length=5)
    >>> tokenizer(inputs)
    <tf.Tensor: shape=(5,), dtype=int32,
        numpy=array([[ 105,   32,  108,  105,  107],
       [2350, 2376, 2306,   32, 2325]], dtype=int32)>

    Tokenization with vocabulary_size.
    >>> latin_ext_cutoff = 592
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCharacterTokenizer(
    ...     vocabulary_size=latin_ext_cutoff)
    >>> tokenizer("¿Cómo estás?")
    <tf.Tensor: shape=(10,), dtype=int32,
    numpy=array([191,  99, 243, 109, 111,  32, 101, 115, 116, 225, 115,  63],
    dtype=int32)>
    >>> tokenizer("आप कैसे हैं")
    <tf.Tensor: shape=(11,), dtype=int32,
    numpy=array([591, 591,  32, 591, 591, 591, 591,  32, 591, 591, 591],
    dtype=int32)>

    Detokenization.
    >>> inputs = tf.constant([110, 105, 110, 106,  97], dtype=tf.int32)
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCharacterTokenizer()
    >>> tokenizer.detokenize(inputs)
    <tf.Tensor: shape=(), dtype=string, numpy=b'ninja'>

    Detokenization with padding.
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCharacterTokenizer(
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
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCharacterTokenizer(
    ...     errors="replace", replacement_char=88)
    >>> tokenizer.detokenize(inputs).numpy().decode('utf-8')
    'niXnja'
    """

    def __init__(
        self,
        sequence_length: int = None,
        lowercase: bool = True,
        normalization_form: str = None,
        errors: str = "replace",
        replacement_char: int = 65533,
        input_encoding: str = "UTF-8",
        output_encoding: str = "UTF-8",
        vocabulary_size: int = None,
        **kwargs,
    ) -> None:
        # Check dtype and provide a default.
        if "dtype" not in kwargs or kwargs["dtype"] is None:
            kwargs["dtype"] = tf.int32
        else:
            dtype = tf.dtypes.as_dtype(kwargs["dtype"])
            if not dtype.is_integer:
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

        super().__init__(**kwargs)

        self.sequence_length = sequence_length
        self.lowercase = lowercase
        self.normalization_form = normalization_form
        self.errors = errors
        self.replacement_char = replacement_char
        self.input_encoding = input_encoding
        self.output_encoding = output_encoding
        self.vocabulary_size = vocabulary_size

    def get_config(self) -> Dict[str, Any]:
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
                "vocabulary_size": self.vocabulary_size,
            }
        )
        return config

    def vocabulary_size(self) -> int:
        """Get the size of the tokenizer vocabulary. None implies no vocabulary
        size was provided"""
        return self.vocabulary_size

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
        if self.vocabulary_size:
            tokens = tf.clip_by_value(tokens, 0, self.vocabulary_size - 1)

        return tokens

    def detokenize(self, inputs):
        inputs = tf.ragged.boolean_mask(inputs, tf.not_equal(inputs, 0))
        encoded_string = tf.strings.unicode_encode(
            inputs,
            errors=self.errors,
            replacement_char=self.replacement_char,
            output_encoding=self.output_encoding,
        )
        return encoded_string
