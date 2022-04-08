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
from typing import Iterable
from typing import List
from typing import Union

import tensorflow as tf
import tensorflow_text as tf_text

from keras_nlp.tokenizers import tokenizer

# Matches whitespace and control characters.
WHITESPACE_REGEX = r"|".join(
    [
        r"\s",
        # Invisible control characters
        r"\p{Cc}",
        r"\p{Cf}",
    ]
)

# Matches punctuation compatible with the original bert implementation.
PUNCTUATION_REGEX = r"|".join(
    [
        # Treat all non-letter/number ASCII as punctuation.
        # Characters such as "^", "$", and "`" are not in the Unicode
        # Punctuation class but we treat them as punctuation anyways.
        r"[!-/]",
        r"[:-@]",
        r"[\[-`]",
        r"[{-~]",
        # Unicode punctuation class.
        r"[\p{P}]",
        # More unicode ranges.
        r"[\x{4E00}-\x{9FFF}]",
        r"[\x{3400}-\x{4DBF}]",
        r"[\x{20000}-\x{2A6DF}]",
        r"[\x{2A700}-\x{2B73F}]",
        r"[\x{2B740}-\x{2B81F}]",
        r"[\x{2B820}-\x{2CEAF}]",
        r"[\x{F900}-\x{FAFF}]",
        r"[\x{2F800}-\x{2FA1F}]",
    ]
)

# Matches both whitespace and punctuation.
WHITESPACE_AND_PUNCTUATION_REGEX = r"|".join(
    [
        WHITESPACE_REGEX,
        PUNCTUATION_REGEX,
    ]
)


class UnicodeCharacterTokenizer(tokenizer.Tokenizer):
    """A unicode character tokenizer layer.

    This tokenizer is a vocabulary free tokenizer which tokenizes text as 
    unicode characters codepoints.

    Args:
        lowercase: If true, the input text will be first lowered before
            tokenization.
        sequence_length: If set, the output will be converted to a dense
            tensor and padded/trimmed so all outputs are of sequence_length.
        strip_accents: If true, all accent marks will be removed from text
            before tokenization.
        normalization_form: One of the following string values (None, 'NFC',
            'NFKC', 'NFD', 'NFKD'). If set will normalize unicode to the given 
            form before tokenizing.
        errors: One of ('replace', 'remove', 'strict'). Specifies the 
            `detokenize()` behavior when an invalid codepoint is encountered. 
            (same behavior as 
            https://www.tensorflow.org/api_docs/python/tf/strings/unicode_transcode)
        replacement_char: The unicode codepoint to use in place of invalid
            codepoints. Defaults to 65533 (U+FFFD).
        input_encoding: The encoding of the input text. Defaults to "utf-8".
        output_encoding: The encoding of the output text. Defaults to "utf-8".

    Examples:

    Basic Usage.
    >>> inputs = "Unicode Tokenizer"
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCharacterTokenizer()
    >>> tokenizer(inputs)
    <tf.Tensor: shape=(17,), dtype=int32, numpy=
    array([117, 110, 105,  99, 111, 100, 101,  32, 116, 111, 107, 101, 110,
        105, 122, 101, 114], dtype=int32)>

    Ragged outputs.
    >>> inputs = ["Ninja", "Samurai"]
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCharacterTokenizer()
    >>> tokenizer(inputs)
    <tf.RaggedTensor [[110, 105, 110, 106, 97], 
        [115, 97, 109, 117, 114, 97, 105]]>

    Dense outputs.
    >>> inputs = ["Ninja", "Samurai"]
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCharacterTokenizer(
        sequence_length=8)
    >>> tokenizer(inputs)
    <tf.Tensor: shape=(2, 8), dtype=int32, numpy=
    array([[110, 105, 110, 106,  97,   0,   0,   0],
        [115,  97, 109, 117, 114,  97, 105,   0]], dtype=int32)>

    Tokenize first, then batch the dataset up.
    >>> inputs = ["Ninja", "Samurai"]
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCharacterTokenizer()
    >>> ds = tf.data.Dataset.from_tensor_slices(inputs)
    >>> ds = ds.map(tokenizer)
    >>> ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(2))
    >>> ds.take(1).get_single_element()
    <tf.RaggedTensor [[110, 105, 110, 106, 97], 
        [115, 97, 109, 117, 114, 97, 105]]>

    Batch up the inputs and then tokenize.
    >>> inputs = ["Ninja", "Samurai"]
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCharacterTokenizer()
    >>> ds = tf.data.Dataset.from_tensor_slices(inputs)
    >>> ds = ds.batch(2).map(tokenizer)
    >>> ds.take(1).get_single_element()
    <tf.RaggedTensor [[110, 105, 110, 106, 97], 
        [115, 97, 109, 117, 114, 97, 105]]>

    Tokenize first, then batch the dataset up for Dense Outputs 
    (`sequence_length` provided).
    >>> inputs = ["Ninja", "Samurai"]
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCharacterTokenizer(
        sequence_length=5)
    >>> ds = tf.data.Dataset.from_tensor_slices(inputs)
    >>> ds = ds.map(tokenizer)
    >>> ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(2))
    >>> ds.take(1).get_single_element()
    <tf.Tensor: shape=(2, 5), dtype=int32, numpy=
    array([[110, 105, 110, 106,  97],
        [115,  97, 109, 117, 114]], dtype=int32)>

    Batch up the inputs and then tokenize for Dense Outputs 
    (`sequence_length` provided).
    >>> inputs = ["Ninja", "Samurai"]
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCharacterTokenizer(
        sequence_length=5)
    >>> ds = tf.data.Dataset.from_tensor_slices(inputs)
    >>> ds = ds.batch(2).map(tokenizer)
    >>> ds.take(1).get_single_element()
    <tf.Tensor: shape=(2, 5), dtype=int32, numpy=
    array([[110, 105, 110, 106,  97],
        [115,  97, 109, 117, 114]], dtype=int32)>

    Tokenization Showcasing Truncation of Long Sequences.
    >>> inputs = "I Like to Travel a Lot"
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCharacterTokenizer(
        sequence_length=5)
    >>> tokenizer(inputs)
    <tf.Tensor: shape=(5,), dtype=int32, 
        numpy=array([105,  32, 108, 105, 107], dtype=int32)>

    Detokenization.
    >>> inputs = tf.constant([110, 105, 110, 106,  97], dtype=tf.int32)
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCharacterTokenizer()
    >>> tokenizer.detokenize(inputs)
    <tf.Tensor: shape=(), dtype=string, numpy=b'ninja'>

    Detokenization while showcasing padded characters being removed
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCharacterTokenizer(sequence_length=7)
    >>> dataset = tf.data.Dataset.from_tensor_slices(["a b c", "b c", "a"])
    >>> dataset = dataset.map(tokenizer)
    >>> dataset.take(1).get_single_element()
    <tf.Tensor: shape=(7,), dtype=int32, 
        numpy=array([97, 32, 98, 32, 99,  0,  0], dtype=int32)>
    >>> detokunbatched = dataset.map(tokenizer.detokenize)
    >>> detokunbatched = dataset.map(tokenizer.detokenize)
    >>> detokunbatched.take(1).get_single_element()
    <tf.Tensor: shape=(), dtype=string, numpy=b'a b c'>

    Detokenization with invalid bytes.
    >>> # The 10000000 in the inputs tensor below is an invalid valye
    >>> # Hence it replaces to the replacement_char 75 which represents 'K'
    >>> inputs = tf.constant([110, 105, 10000000, 110, 106,  97], dtype=tf.int32)
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCharacterTokenizer(
    ...     errors="replace", replacement_char=75)
    >>> tokenizer.detokenize(inputs).numpy().decode('utf-8')
    'niKnja'
    """

    def __init__(
        self,
        sequence_length: int = None,
        lowercase: bool = True,
        normalization_form: str = None,
        errors: str = "replace",
        replacement_char: int = 65533,
        input_encoding: str = "utf8",
        output_encoding: str = "UTF-8",
        **kwargs,
    ) -> None:
        # Check dtype and provide a default.
        if "dtype" not in kwargs or kwargs["dtype"] is None:
            kwargs["dtype"] = tf.int32
        else:
            dtype = tf.dtypes.as_dtype(kwargs["dtype"])
            if not dtype.is_integer and dtype != tf.string:
                raise ValueError(
                    "Output dtype must be an integer type of a string. "
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

        super().__init__(**kwargs)

        self.sequence_length = sequence_length
        self.lowercase = lowercase
        self.normalization_form = normalization_form
        self.errors = errors
        self.replacement_char = replacement_char
        self.input_encoding = input_encoding
        self.output_encoding = output_encoding

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
            }
        )
        return config
        
    def tokenize(self, inputs):

        if not isinstance(inputs, (tf.Tensor, tf.RaggedTensor)):
            inputs = tf.convert_to_tensor(inputs)

        scalar_input = inputs.shape.rank == 0
        if scalar_input:
            inputs = tf.expand_dims(inputs, 0)

        # Optionally Lowercase the Text
        if self.lowercase:
            inputs = tf_text.case_fold_utf8(inputs)

        # Optionally Normalize the Text to a given form
        if self.normalization_form:
            if (self.input_encoding != "utf8"):
                raise ValueError("Normalization Forms are Only Supported for Input Encoding utf-8")
            else:
                inputs = tf_text.normalize_utf8(inputs, self.normalization_form)

        # Apply Unicode Decoder
        tokens = tf.strings.unicode_decode(inputs, errors=self.errors, 
            replacement_char=self.replacement_char, 
            input_encoding=self.input_encoding)

        if self.sequence_length:
            output_shape = tokens.shape.as_list()
            output_shape[-1] = self.sequence_length
            tokens = tokens.to_tensor(shape=output_shape)

        if scalar_input:
            tokens = tf.squeeze(tokens, 0)
        return tokens

    def detokenize(self, inputs):
        encoded_string = tf.strings.unicode_encode(inputs, errors=self.errors, 
            replacement_char=self.replacement_char, 
            output_encoding=self.output_encoding)
        encoded_string_with_removed_null = tf.strings.regex_replace(
            encoded_string, r"\x00+$", "")
        return encoded_string_with_removed_null

