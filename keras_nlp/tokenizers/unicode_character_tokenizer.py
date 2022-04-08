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

    Ragged outputs.
    >>> inputs = [["a b c", "b c", "a b"]]
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCharacterTokenizer()
    >>> tokenizer(inputs)
    <tf.RaggedTensor [[97, 32, 98, 32, 99], [98, 32, 99], [97, 32, 98]]>

    Dense outputs.
    >>> inputs = [["a b c", "b c", "a b"]]
    >>> tokenizer = keras_nlp.tokenizers.UnicodeCharacterTokenizer(
    ...     sequence_length=5)
    >>> tokenizer(inputs)
    tf.Tensor([[97 32 98 32 99]
        [98 32 99  0  0]
        [97 32 98  0  0]], shape=(3, 5), dtype=int32)

    String output.
    >>> vocab = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
    >>> inputs = ["The quick brown fox."]
    >>> tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    ...     vocabulary=vocab, dtype="string")
    >>> tokenizer(inputs)
    <tf.RaggedTensor [[b'the', b'qu', b'##ick', b'br', b'##own', b'fox', b'.']]>

    Detokenization.
    >>> vocab = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
    >>> inputs = ["The quick brown fox."]
    >>> tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(vocabulary=vocab)
    >>> tokenizer.detokenize(tokenizer.tokenize(inputs))
    <tf.Tensor: shape=(1,), dtype=string,
        numpy=array([b"the quick brown fox ."], dtype=object)>

    Custom splitting.
    >>> vocab = ["[UNK]", "fox", ","]
    >>> inputs = ["fox,,fox,fox"]
    >>> tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(vocabulary=vocab,
    ...     split_pattern=",", keep_pattern=",", dtype='string')(inputs)
    <tf.RaggedTensor [[b'fox', b',', b',', b'fox', b',', b'fox']]>
    >>> keras_nlp.tokenizers.WordPieceTokenizer(vocabulary=vocab,
    ...     split_pattern=",", keep_pattern="", dtype='string')(inputs)
    <tf.RaggedTensor [[b'fox', b'fox', b'fox']]>
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

        super().__init__(**kwargs)

        self._sequence_length = sequence_length
        self._lowercase = lowercase
        self._normalization_form = normalization_form
        self._errors = errors
        self._replacement_char = replacement_char
        self._input_encoding = input_encoding
        self._output_encoding = output_encoding

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                # Ideally a vocabulary would be saved as a plain text asset in
                # the saved model. We have no good way to support this
                # currently, so we save the vocabulary in the config.
                "vocabulary": self._vocab,
                "sequence_length": self._sequence_length,
                "lowercase": self._lowercase,
                "normalization_form": self._normalization_form,
                "errors": self._errors,
                "replacement_char": self._replacement_char,
                "input_encoding": self._input_encoding,
                "output_encoding": self._output_encoding,
            }
        )
        return config
        
    def tokenize(self, inputs):
        # scalar_input = inputs.shape.rank == 0
        # if scalar_input:
        #     inputs = tf.expand_dims(inputs, 0)
        # Optionally lowercase and normalize the input.
        inputs = tf.convert_to_tensor(inputs)
        print(inputs)
        print(inputs.shape)
        print(inputs.ndim)
        if self._lowercase:
            inputs = tf_text.case_fold_utf8(inputs)
        if self._normalization_form:
            if (self._input_encoding != "utf8"):
                raise ValueError("Normalization Forms are Only Supported for Input Encoding utf-8")
            else:
                inputs = tf_text.normalize_utf8(inputs, self._normalization_form)
        # Apply Unicode Decoder
        print(inputs)
        tokens = tf.strings.unicode_decode(inputs, errors=self._errors, 
            replacement_char=self._replacement_char, 
            input_encoding=self._input_encoding)
        print(tokens)
        # Convert to a dense output if `sequence_length` is set.
        # if self._sequence_length:
        #     output_shape = tokens.shape.as_list()
        #     output_shape[-1] = self._sequence_length
        #     # tokens = tokens.to_tensor(shape=output_shape)
        #     tf.reshape(tokens, output_shape)
        return tokens
        # if scalar_input:
        #     tokens = tf.squeeze(tokens, 0)
        # return tokens

    def detokenize(self, inputs):
        return tf.strings.unicode_encode(inputs, errors=self._errors, 
            replacement_char=self._replacement_char, 
            output_encoding=self._output_encoding)


