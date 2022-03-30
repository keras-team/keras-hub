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


class WordPieceTokenizer(tokenizer.Tokenizer):
    """A word piece tokenizer layer.

    This layer provides an efficient, in graph, implementation of the WordPiece
    algorithm used by BERT and other models.

    To make this layer more useful out of the box, the layer will pre-tokenize
    the input, which will optionally lower-case, strip accents, and split the
    input on whitespace and punctuation. Each of these pre-tokenization steps is
    not reversible. The `detokenize` method will join words with a space, and
    will not invert `tokenize` exactly.

    If a more custom pre-tokenization step is desired, the layer can be
    configured to apply only the strict WordPiece algorithm by passing
    `lowercase=False`, `strip_accents=False` and `split_pattern=None`. In
    this case, inputs should be pre-split string tensors or ragged tensors.

    By default, the layer will output a `tf.RaggedTensor` where the last
    dimension of the output is ragged after whitespace splitting and sub-word
    tokenizing. If `sequence_length` is set, the layer will output a dense
    `tf.Tensor` where all inputs have been padded or truncated to
    `sequence_length`. The output dtype can be controlled via the `dtype`
    argument, which should be either an integer or string type.

    Args:
        vocabulary: A list of strings or a string string filename path. If
            passing a list, each element of the list should be a single word
            piece token string. If passing a filename, the file should be a
            plain text file containing a single word piece token per line.
        sequence_length: If set, the output will be converted to a dense
            tensor and padded/trimmed so all outputs are of sequence_length.
        lowercase: If true, the input text will be first lowered before
            tokenization.
        strip_accents: If true, all accent marks will be removed from text
            before tokenization.
        split_pattern: A regex pattern to match delimiters to split, or None
            indicating that the input is pre-split and no splitting should be
            performed. By default, all whitespace and punctuation marks will
            be split on.
        keep_pattern: A regex pattern of delimiters contained in the
            `split_pattern` of delimeters that should be kept as independent
            tokens. By default, all punctuation marks will be kept as tokens.
        suffix_indicator: The characters prepended to a wordpiece to indicate
            that it is a suffix to another subword.
        oov_token: The string value to substitute for an unknown token. It
            must be included in the vocab.

    References:
        - [Schuster and Nakajima, 2012](https://research.google/pubs/pub37842/)
        - [Song et al., 2020](https://arxiv.org/abs/2012.15524)

    Examples:

    Ragged outputs.
    >>> vocab = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
    >>> inputs = ["The quick brown fox."]
    >>> tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(vocabulary=vocab)
    >>> tokenizer(inputs)
    <tf.RaggedTensor [[1, 2, 3, 4, 5, 6, 7]]>

    Dense outputs.
    >>> vocab = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
    >>> inputs = ["The quick brown fox."]
    >>> tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    ...     vocabulary=vocab, sequence_length=10)
    >>> tokenizer(inputs)
    <tf.Tensor: shape=(1, 10), dtype=int32,
        numpy=array([[1, 2, 3, 4, 5, 6, 7, 0, 0, 0]]>

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
        vocabulary: Union[Iterable[str], str] = None,
        sequence_length: int = None,
        lowercase: bool = True,
        strip_accents: bool = True,
        split_pattern: str = WHITESPACE_AND_PUNCTUATION_REGEX,
        keep_pattern: str = PUNCTUATION_REGEX,
        suffix_indicator: str = "##",
        oov_token: str = "[UNK]",
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

        if isinstance(vocabulary, str):
            self._vocab = [
                line.rstrip() for line in tf.io.gfile.GFile(vocabulary)
            ]
        elif isinstance(vocabulary, Iterable):
            # Make a copy.
            self._vocab = list(vocabulary)
        else:
            raise ValueError(
                "Vocabulary must be an file path or list of terms. "
                f"Received: vocabulary={vocabulary}"
            )

        self._sequence_length = sequence_length
        self._lowercase = lowercase
        self._strip_accents = strip_accents
        self._split_pattern = split_pattern
        self._keep_pattern = keep_pattern
        self._suffix_indicator = suffix_indicator
        self._oov_token = oov_token

        self._fast_word_piece = tf_text.FastWordpieceTokenizer(
            vocab=self._vocab,
            token_out_type=self.compute_dtype,
            suffix_indicator=suffix_indicator,
            unknown_token=oov_token,
            no_pretokenization=True,
            support_detokenization=True,
        )

    def get_vocabulary(self) -> List[str]:
        """Get the tokenizer vocabulary as a list of strings tokens."""
        return self._vocab

    def vocabulary_size(self) -> int:
        """Get the size of the tokenizer vocabulary."""
        return len(self._vocab)

    def id_to_token(self, id: int) -> str:
        """Convert an integer id to a string token."""
        return self._vocab[id]

    def token_to_id(self, token: str) -> int:
        """Convert a string token to an integer id."""
        # This will be slow, but keep memory usage down compared to building a
        # dict. Assuming the main use case is looking up a few special tokens
        # early in the vocab, this should be fine.
        return self._vocab.index(token)

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
                "strip_accents": self._strip_accents,
                "split_pattern": self._split_pattern,
                "keep_pattern": self._keep_pattern,
                "suffix_indicator": self._suffix_indicator,
                "oov_token": self._oov_token,
            }
        )
        return config

    def tokenize(self, inputs):
        # Optionally normalize and split inputs.
        if self._lowercase:
            inputs = tf_text.case_fold_utf8(inputs)
        if self._strip_accents:
            # Normalize unicode to NFD, which splits out accent mark characters.
            inputs = tf_text.normalize_utf8(inputs, "NFD")
            # Remove the accent marks.
            inputs = tf.strings.regex_replace(inputs, r"\p{Mn}", "")
        if self._split_pattern:
            inputs = tf_text.regex_split(
                inputs,
                delim_regex_pattern=self._split_pattern,
                keep_delim_regex_pattern=self._keep_pattern,
            )

        # Apply word piece and coerce shape for outputs.
        tokens = self._fast_word_piece.tokenize(inputs)
        # By default tf.text tokenizes text with two ragged dimensions (one for
        # split words and one for split subwords). We will collapse to a single
        # ragged dimension which is a better out of box default.
        tokens = tokens.merge_dims(-2, -1)
        # Convert to a dense output if `sequence_length` is set.
        if self._sequence_length:
            output_shape = tokens.shape.as_list()
            output_shape[-1] = self._sequence_length
            tokens = tokens.to_tensor(shape=output_shape)
        return tokens

    def detokenize(self, inputs):
        return self._fast_word_piece.detokenize(inputs)
