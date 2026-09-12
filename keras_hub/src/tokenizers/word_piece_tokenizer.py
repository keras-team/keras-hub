import functools
import os
import re
import unicodedata
from typing import Iterable

import keras
import numpy as np
import regex
from keras.src.saving import serialization_lib

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.tokenizers import tokenizer
from keras_hub.src.utils.tensor_utils import canonicalize_python_string_inputs
from keras_hub.src.utils.tensor_utils import canonicalize_python_token_inputs
from keras_hub.src.utils.tensor_utils import casefold_utf8
from keras_hub.src.utils.tensor_utils import convert_to_numpy
from keras_hub.src.utils.tensor_utils import convert_to_ragged_batch
from keras_hub.src.utils.tensor_utils import in_tf_function
from keras_hub.src.utils.tensor_utils import is_int_dtype
from keras_hub.src.utils.tensor_utils import is_string_dtype
from keras_hub.src.utils.tensor_utils import preprocessing_function

try:
    import tensorflow as tf
except ImportError:
    tf = None
try:
    import tensorflow_text as tf_text
except ImportError:
    tf_text = None

VOCAB_FILENAME = "vocabulary.txt"

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
    ]
)

# Matches CJK characters. Obtained from
# https://github.com/google-research/bert/blob/master/tokenization.py#L251.
CJK_REGEX = r"|".join(
    [
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

# Matches punctuation and CJK characters.
PUNCTUATION_AND_CJK_REGEX = r"|".join(
    [
        PUNCTUATION_REGEX,
        CJK_REGEX,
    ]
)

# Matches whitespace, punctuation, and CJK characters.
WHITESPACE_PUNCTUATION_AND_CJK_REGEX = r"|".join(
    [
        WHITESPACE_AND_PUNCTUATION_REGEX,
        CJK_REGEX,
    ]
)


# Python `regex` equivalents of the RE2 patterns above, used by the Python
# (TensorFlow free) path. Note that RE2's `\s` only matches ASCII whitespace.
PY_WHITESPACE_REGEX = r"|".join(
    [
        r"[ \t\n\r\f]",
        r"\p{Cc}",
        r"\p{Cf}",
    ]
)
PY_PUNCTUATION_REGEX = r"|".join(
    [
        r"[!-/]",
        r"[:-@]",
        r"[\[-`]",
        r"[{-~]",
        r"[\p{P}]",
    ]
)
PY_CJK_REGEX = r"|".join(
    [
        r"[\u4E00-\u9FFF]",
        r"[\u3400-\u4DBF]",
        r"[\U00020000-\U0002A6DF]",
        r"[\U0002A700-\U0002B73F]",
        r"[\U0002B740-\U0002B81F]",
        r"[\U0002B820-\U0002CEAF]",
        r"[\uF900-\uFAFF]",
        r"[\U0002F800-\U0002FA1F]",
    ]
)

# `tf_text.FastWordpieceTokenizer` maps words longer than this many bytes to
# the unknown token.
MAX_BYTES_PER_WORD = 100


@functools.lru_cache(maxsize=32)
def _get_split_patterns(split_on_cjk, special_tokens):
    """Compiled `(delimiter, keep_delimiter)` regexes for word splitting."""
    if split_on_cjk:
        split_pattern = r"|".join(
            [PY_WHITESPACE_REGEX, PY_PUNCTUATION_REGEX, PY_CJK_REGEX]
        )
        keep_split_pattern = r"|".join([PY_PUNCTUATION_REGEX, PY_CJK_REGEX])
    else:
        split_pattern = r"|".join([PY_WHITESPACE_REGEX, PY_PUNCTUATION_REGEX])
        keep_split_pattern = PY_PUNCTUATION_REGEX
    if special_tokens:
        special_tokens_pattern = get_special_tokens_pattern(special_tokens)
        split_pattern = r"|".join([special_tokens_pattern, split_pattern])
        keep_split_pattern = r"|".join(
            [special_tokens_pattern, keep_split_pattern]
        )
    return regex.compile(split_pattern), regex.compile(keep_split_pattern)


def _regex_split(text, delimiter_pattern, keep_pattern):
    """Python equivalent of `tf_text.regex_split` for a single string."""
    words = []
    position = 0
    for match in delimiter_pattern.finditer(text):
        if match.start() > position:
            words.append(text[position : match.start()])
        if keep_pattern.fullmatch(match.group()):
            words.append(match.group())
        position = match.end()
    if position < len(text):
        words.append(text[position:])
    return words


def pretokenize_python(
    text,
    lowercase=False,
    strip_accents=True,
    split=True,
    split_on_cjk=True,
    special_tokens=None,
):
    """Python equivalent of `pretokenize` for a single string.

    Args:
        text: str. Input to be pretokenized.
        lowercase: bool. If True, the input text will be lowercased (with
            unicode case folding) before tokenization.
        strip_accents: bool. If `True`, all accent marks will be removed from
            text before tokenization.
        split: bool. If `True`, input will be split on whitespace and
            punctuation marks, and all punctuation marks will be kept as
            tokens. If `False`, `text` is treated as a single whole word.
        split_on_cjk: bool. If `True`, input will be split on CJK characters.
            Only applicable when `split` is `True`.
        special_tokens: list of strings. Special tokens that will never be
            split or lowercased.

    Returns:
        A list of pre-tokenized words.
    """
    if strip_accents:
        # Normalize unicode to NFD, which splits out accent mark characters,
        # then remove the accent marks.
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    special_tokens = tuple(special_tokens) if special_tokens else ()
    if split:
        delimiter_pattern, keep_pattern = _get_split_patterns(
            split_on_cjk, special_tokens
        )
        words = _regex_split(text, delimiter_pattern, keep_pattern)
    else:
        words = [text]
    if lowercase:
        # Do not lowercase special tokens. They often contain capital
        # letters, e.g. `"[CLS]"`.
        words = [w if w in special_tokens else casefold_utf8(w) for w in words]
    return words


def get_special_tokens_pattern(special_tokens):
    if special_tokens is None or len(special_tokens) == 0:
        return None
    return r"|".join([re.escape(token) for token in special_tokens])


def pretokenize(
    text,
    lowercase=False,
    strip_accents=True,
    split=True,
    split_on_cjk=True,
    special_tokens_pattern=None,
):
    """Helper function that takes in a dataset element and pretokenizes it.

    Args:
        text: `tf.Tensor` or `tf.RaggedTensor`. Input to be pretokenized.
        lowercase: bool. If True, the input text will be
            lowercased before tokenization. Defaults to `True`.
        strip_accents: bool. If `True`, all accent marks will
            be removed from text before tokenization. Defaults to `True`.
        split: bool. If `True`, input will be split on
            whitespace and punctuation marks, and all punctuation marks will be
            kept as tokens. If `False`, input should be split ("pre-tokenized")
            before calling the tokenizer, and passed as a dense or ragged tensor
            of whole words. Defaults to `True`.
        split_on_cjk: bool. If `True`, input will be split
            on CJK characters, i.e., Chinese, Japanese, Korean and Vietnamese
            characters (https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)).
            Note that this is applicable only when `split` is `True`. Defaults
            to `True`.
        special_tokens_pattern: str. A regex pattern that contain the
            special tokens that will never be split during the word-level
            splitting applied before the word-peice encoding. This can be used
            to ensure special tokens map to unique indices in the vocabulary,
            even if these special tokens contain splittable characters such as
            punctuation.

    Returns:
        A tensor containing the pre-processed and pre-tokenized `text`.
    """
    # Check for correct types.
    if not is_string_dtype(text.dtype):
        raise ValueError(
            "The dataset elements in `data` must have string dtype. "
            f"Received: {text.dtype}."
        )
    # Preprocess, lowercase, strip and split input data.
    if text.shape.rank == 0:
        text = tf.expand_dims(text, 0)
    if split_on_cjk and split:
        text = tf.strings.regex_replace(text, CJK_REGEX, r" \0 ")
    if strip_accents:
        # Normalize unicode to NFD, which splits out accent mark characters.
        text = tf_text.normalize_utf8(text, "NFD")
        # Remove the accent marks.
        text = tf.strings.regex_replace(text, r"\p{Mn}", "")
    if split:
        if split_on_cjk:
            split_pattern = WHITESPACE_PUNCTUATION_AND_CJK_REGEX
            keep_split_pattern = PUNCTUATION_AND_CJK_REGEX
        else:
            split_pattern = WHITESPACE_AND_PUNCTUATION_REGEX
            keep_split_pattern = PUNCTUATION_REGEX
        if special_tokens_pattern is not None:
            # the idea here is to pass the special tokens regex to the split
            # function as delimiter regex pattern, so the input will be splitted
            # by them, but also the function will treat each one of them as one
            # entity that shouldn't be splitted even if they have other
            # delimiter regex pattern inside them. then pass the special tokens
            # regex also as keep delimiter regex pattern, so they will
            # not be removed.
            split_pattern = r"|".join(
                [
                    special_tokens_pattern,
                    split_pattern,
                ]
            )
            keep_split_pattern = r"|".join(
                [special_tokens_pattern, keep_split_pattern]
            )
        text = tf_text.regex_split(
            text,
            delim_regex_pattern=split_pattern,
            keep_delim_regex_pattern=keep_split_pattern,
        )
    if lowercase:
        if special_tokens_pattern is not None:
            # Do not lowercase special tokens in string space. They often
            # contain capital letters, e.g. `"[CLS]"`.
            mask = (
                tf.strings.regex_replace(text, special_tokens_pattern, "६")
                == "६"
            )
            text = tf.where(mask, text, tf_text.case_fold_utf8(text))
        else:
            text = tf_text.case_fold_utf8(text)

    return text


@keras_hub_export("keras_hub.tokenizers.WordPieceTokenizer")
class WordPieceTokenizer(tokenizer.Tokenizer):
    """A WordPiece tokenizer layer.

    This layer provides an efficient, in graph, implementation of the WordPiece
    algorithm used by BERT and other models.

    To make this layer more useful out of the box, the layer will pre-tokenize
    the input, which will optionally lower-case, strip accents, and split the
    input on whitespace and punctuation. Each of these pre-tokenization steps is
    not reversible. The `detokenize` method will join words with a space, and
    will not invert `tokenize` exactly.

    If a more custom pre-tokenization step is desired, the layer can be
    configured to apply only the strict WordPiece algorithm by passing
    `lowercase=False`, `strip_accents=False` and `split=False`. In
    this case, inputs should be pre-split string tensors or ragged tensors.

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

    The output dtype can be controlled via the `dtype` argument, which should
    be either an integer or string type.

    Args:
        vocabulary: A list of strings or a string filename path. If
            passing a list, each element of the list should be a single
            WordPiece token string. If passing a filename, the file should be a
            plain text file containing a single WordPiece token per line.
        sequence_length: int. If set, the output will be converted to a dense
            tensor and padded/trimmed so all outputs are of sequence_length.
        lowercase: bool. If `True`, the input text will be
            lowercased before tokenization. Defaults to `False`.
        strip_accents: bool. If `True`, all accent marks will
            be removed from text before tokenization. Defaults to `False`.
        split: bool. If `True`, input will be split on
            whitespace and punctuation marks, and all punctuation marks will be
            kept as tokens. If `False`, input should be split ("pre-tokenized")
            before calling the tokenizer, and passed as a dense or ragged tensor
            of whole words. Defaults to `True`.
        split_on_cjk: bool. If True, input will be split
            on CJK characters, i.e., Chinese, Japanese, Korean and Vietnamese
            characters (https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)).
            Note that this is applicable only when `split` is True.
            Defaults to `True`.
        suffix_indicator: str. The characters prepended to a
            WordPiece to indicate that it is a suffix to another subword.
            E.g. "##ing". Defaults to `"##"`.
        oov_token: str. The string value to substitute for
            an unknown token. It must be included in the vocab.
            Defaults to `"[UNK]"`.
        special_tokens_in_strings: bool. A bool to indicate if the tokenizer
            should expect special tokens in input strings that should be
            tokenized and mapped correctly to their ids. Defaults to False.

    References:
     - [Schuster and Nakajima, 2012](https://research.google/pubs/pub37842/)
     - [Song et al., 2020](https://arxiv.org/abs/2012.15524)

    Examples:

    Ragged outputs.
    >>> vocab = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
    >>> inputs = "The quick brown fox."
    >>> tokenizer = keras_hub.tokenizers.WordPieceTokenizer(
    ...     vocabulary=vocab,
    ...     lowercase=True,
    ... )
    >>> outputs = tokenizer(inputs)
    >>> np.array(outputs)
    array([1, 2, 3, 4, 5, 6, 7], dtype=int32)

    Dense outputs.
    >>> vocab = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
    >>> inputs = ["The quick brown fox."]
    >>> tokenizer = keras_hub.tokenizers.WordPieceTokenizer(
    ...     vocabulary=vocab,
    ...     sequence_length=10,
    ...     lowercase=True,
    ... )
    >>> outputs = tokenizer(inputs)
    >>> np.array(outputs)
    array([[1, 2, 3, 4, 5, 6, 7, 0, 0, 0]], dtype=int32)

    String output.
    >>> vocab = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
    >>> inputs = "The quick brown fox."
    >>> tokenizer = keras_hub.tokenizers.WordPieceTokenizer(
    ...     vocabulary=vocab,
    ...     lowercase=True,
    ...     dtype="string",
    ... )
    >>> tokenizer(inputs)
    ['the', 'qu', '##ick', 'br', '##own', 'fox', '.']

    Detokenization.
    >>> vocab = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
    >>> inputs = "The quick brown fox."
    >>> tokenizer = keras_hub.tokenizers.WordPieceTokenizer(
    ...     vocabulary=vocab,
    ...     lowercase=True,
    ... )
    >>> tokenizer.detokenize(tokenizer.tokenize(inputs))
    'the quick brown fox .'

    Custom splitting.
    >>> vocab = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
    >>> inputs = "The$quick$brown$fox"
    >>> tokenizer = keras_hub.tokenizers.WordPieceTokenizer(
    ...     vocabulary=vocab,
    ...     split=False,
    ...     lowercase=True,
    ...     dtype='string',
    ... )
    >>> split_inputs = tf.strings.split(inputs, sep="$")
    >>> tokenizer(split_inputs)
    ['the', 'qu', '##ick', 'br', '##own', 'fox']
    """

    def __init__(
        self,
        vocabulary=None,
        sequence_length=None,
        lowercase=False,
        strip_accents=False,
        split=True,
        split_on_cjk=True,
        suffix_indicator="##",
        oov_token="[UNK]",
        special_tokens=None,
        special_tokens_in_strings=False,
        dtype="int32",
        **kwargs,
    ) -> None:
        if not is_int_dtype(dtype) and not is_string_dtype(dtype):
            raise ValueError(
                "Output dtype must be an integer type or a string. "
                f"Received: dtype={dtype}"
            )

        _allow_python_workflow = kwargs.pop("_allow_python_workflow", True)
        super().__init__(
            dtype=dtype, _allow_python_workflow=_allow_python_workflow, **kwargs
        )
        if oov_token is None:
            raise ValueError("`oov_token` cannot be None.")

        self.sequence_length = sequence_length
        self.lowercase = lowercase
        self.strip_accents = strip_accents
        self.split = split
        self.split_on_cjk = split_on_cjk
        self.suffix_indicator = suffix_indicator
        self.oov_token = oov_token
        self._init_special_tokens = special_tokens
        self.special_tokens_in_strings = special_tokens_in_strings

        self.set_vocabulary(vocabulary)
        self.file_assets = [VOCAB_FILENAME]

    def save_assets(self, dir_path):
        path = os.path.join(dir_path, VOCAB_FILENAME)
        with open(path, "w", encoding="utf-8") as file:
            for token in self.vocabulary:
                file.write(f"{token}\n")

    def load_assets(self, dir_path):
        path = os.path.join(dir_path, VOCAB_FILENAME)
        self.set_vocabulary(path)

    def set_vocabulary(self, vocabulary):
        """Set the tokenizer vocabulary to a file or list of strings."""
        if vocabulary is None:
            self.vocabulary = None
            self._fast_word_piece = None
            self._token_to_id_map = None
            return

        if isinstance(vocabulary, str):
            if serialization_lib.in_safe_mode():
                raise ValueError(
                    "Requested the loading of a vocabulary file outside of the "
                    "model archive. This carries a potential risk of loading "
                    "arbitrary and sensitive files and thus it is disallowed "
                    "by default. If you trust the source of the artifact, you "
                    "can override this error by passing `safe_mode=False` to "
                    "the loading function, or calling "
                    "`keras.config.enable_unsafe_deserialization()`. "
                    f"Vocabulary file: '{vocabulary}'"
                )
            with open(vocabulary, "r", encoding="utf-8") as file:
                self.vocabulary = [line.rstrip() for line in file]
        elif isinstance(vocabulary, Iterable):
            # Make a defensive copy.
            self.vocabulary = list(vocabulary)
        else:
            raise ValueError(
                "Vocabulary must be an file path or list of terms. "
                f"Received: vocabulary={vocabulary}"
            )

        if self.oov_token not in self.vocabulary:
            raise ValueError(
                f'Cannot find `oov_token="{self.oov_token}"` in the '
                "vocabulary.\n"
                "You can either update the vocabulary to include "
                f'`"{self.oov_token}"`, or pass a different value for '
                "the `oov_token` argument when creating the tokenizer."
            )

        self._token_to_id_map = {
            token: i for i, token in enumerate(self.vocabulary)
        }
        # When using `WordPieceTokenizer` with `tf.data`, the tf-text
        # tokenizer must be built outside the `tf.data` pipeline, so build it
        # eagerly whenever tf-text is available.
        self._fast_word_piece = None
        if tf_text is not None:
            self._set_vocabulary_tf()
        self._update_special_token_ids()

    def _set_vocabulary_tf(self):
        self._fast_word_piece = tf_text.FastWordpieceTokenizer(
            vocab=self.vocabulary,
            token_out_type=self.compute_dtype,
            suffix_indicator=self.suffix_indicator,
            unknown_token=self.oov_token,
            no_pretokenization=True,
            support_detokenization=True,
        )

    def _maybe_initialized_tf(self):
        if self._fast_word_piece is None:
            self._set_vocabulary_tf()

    def get_vocabulary(self):
        """Get the tokenizer vocabulary as a list of strings tokens."""
        self._check_vocabulary()
        return self.vocabulary

    def vocabulary_size(self):
        """Get the integer size of the tokenizer vocabulary."""
        self._check_vocabulary()
        return len(self.vocabulary)

    def id_to_token(self, id):
        """Convert an integer id to a string token."""
        self._check_vocabulary()
        if id >= self.vocabulary_size() or id < 0:
            raise ValueError(
                f"`id` must be in range [0, {self.vocabulary_size() - 1}]. "
                f"Received: {id}"
            )
        return self.vocabulary[id]

    def token_to_id(self, token):
        """Convert a string token to an integer id."""
        self._check_vocabulary()
        if token not in self._token_to_id_map:
            raise ValueError(f"Token '{token}' is not in the vocabulary.")
        return self._token_to_id_map[token]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary": None,  # Save vocabulary via an asset!
                "sequence_length": self.sequence_length,
                "lowercase": self.lowercase,
                "strip_accents": self.strip_accents,
                "split": self.split,
                "suffix_indicator": self.suffix_indicator,
                "oov_token": self.oov_token,
                "special_tokens": self._init_special_tokens,
                "special_tokens_in_strings": self.special_tokens_in_strings,
            }
        )
        return config

    def _check_vocabulary(self):
        if self.vocabulary is None:
            raise ValueError(
                "No vocabulary has been set for WordPieceTokenizer. Make sure "
                "to pass a `vocabulary` argument when creating the layer."
            )

    def _special_tokens_for_splitting(self):
        if not (self.split and self.special_tokens_in_strings):
            return None
        special_tokens = self.special_tokens
        if self._init_special_tokens:
            special_tokens += self._init_special_tokens
        return special_tokens

    @preprocessing_function
    def _tokenize_tf(self, inputs):
        self._maybe_initialized_tf()
        if not isinstance(inputs, tf.RaggedTensor):
            inputs = tf.convert_to_tensor(inputs)
        unbatched = inputs.shape.rank == 0
        # The idea here is to pass the special tokens regex to the split
        # function as delimiter regex pattern, so the input will be splitted
        # by them, but also the function will treat each one of them as one
        # entity that shouldn't be splitted even if they have other delimiter
        # regex pattern inside them. Then pass the special tokens regex also
        # as keep delimiter regex pattern, so they will not be removed.
        pattern = get_special_tokens_pattern(
            self._special_tokens_for_splitting()
        )
        inputs = pretokenize(
            inputs,
            self.lowercase,
            self.strip_accents,
            self.split,
            self.split_on_cjk,
            pattern,
        )

        # Apply WordPiece and coerce shape for outputs.
        tokens = self._fast_word_piece.tokenize(inputs)
        # By default tf.text tokenizes text with two ragged dimensions (one for
        # split words and one for split subwords). We will collapse to a single
        # ragged dimension which is a better out of box default.
        tokens = tokens.merge_dims(-2, -1)

        # Convert to a dense output if `sequence_length` is set.
        if self.sequence_length:
            output_shape = tokens.shape.as_list()
            output_shape[-1] = self.sequence_length
            tokens = tokens.to_tensor(shape=output_shape)
        # Convert to a dense output if input in scalar
        if unbatched:
            tokens = tf.squeeze(tokens, 0)
            tf.ensure_shape(tokens, shape=[self.sequence_length])

        return tokens

    def _tokenize_word_python(self, word):
        """Apply the WordPiece algorithm to a single pre-tokenized word.

        This is a greedy longest-match-first lookup of subwords in the
        vocabulary, where subwords after the first are prefixed with the
        `suffix_indicator`. Words that cannot be fully tokenized, or that are
        longer than `MAX_BYTES_PER_WORD`, map to the `oov_token`.
        """
        if not word:
            return []
        if len(word.encode("utf-8")) > MAX_BYTES_PER_WORD:
            return [self.oov_token]
        tokens = []
        start = 0
        while start < len(word):
            end = len(word)
            current = None
            while start < end:
                subword = word[start:end]
                if start > 0:
                    subword = self.suffix_indicator + subword
                if subword in self._token_to_id_map:
                    current = subword
                    break
                end -= 1
            if current is None:
                return [self.oov_token]
            tokens.append(current)
            start = end
        return tokens

    def _canonicalize_presplit_inputs(self, inputs):
        """Canonicalize inputs for `split=False` to lists of words.

        Returns a tuple `(inputs, batched)`, where `inputs` is a list of
        samples, each a list of pre-split words.
        """
        if tf is not None and isinstance(inputs, tf.RaggedTensor):
            inputs = inputs.to_list()
        elif (
            isinstance(inputs, np.ndarray)
            or keras.ops.is_tensor(inputs)
            or (tf is not None and isinstance(inputs, tf.Tensor))
        ):
            inputs = convert_to_numpy(inputs)
            if inputs.ndim == 2:
                inputs = inputs.tolist()
        if isinstance(inputs, (tuple, list)) and (
            not inputs or isinstance(inputs[0], (tuple, list))
        ):
            samples = []
            for sample in inputs:
                words, _ = canonicalize_python_string_inputs(list(sample))
                samples.append(words)
            return samples, True
        words, _ = canonicalize_python_string_inputs(inputs)
        return [words], False

    def _tokenize_python(self, inputs):
        special_tokens = self._special_tokens_for_splitting()
        if self.split:
            inputs, batched = canonicalize_python_string_inputs(inputs)
            samples = [[text] for text in inputs]
        else:
            samples, batched = self._canonicalize_presplit_inputs(inputs)

        batched_tokens = []
        for sample in samples:
            tokens = []
            for text in sample:
                words = pretokenize_python(
                    text,
                    self.lowercase,
                    self.strip_accents,
                    self.split,
                    self.split_on_cjk,
                    special_tokens,
                )
                for word in words:
                    tokens.extend(self._tokenize_word_python(word))
            if is_int_dtype(self.compute_dtype):
                tokens = [self._token_to_id_map[token] for token in tokens]
            batched_tokens.append(tokens)

        # Convert to a dense output if `sequence_length` is set.
        if self.sequence_length:
            pad_value = 0 if is_int_dtype(self.compute_dtype) else ""
            batched_tokens = [
                tokens[: self.sequence_length]
                + [pad_value] * (self.sequence_length - len(tokens))
                for tokens in batched_tokens
            ]
            if is_int_dtype(self.compute_dtype):
                # Dense int outputs are arrays, so that direct calls return
                # backend tensors and Grain pipelines return NumPy.
                batched_tokens = np.array(
                    batched_tokens, dtype=self.compute_dtype
                )

        if not batched:
            batched_tokens = batched_tokens[0]
        return batched_tokens

    def tokenize(self, inputs):
        self._check_vocabulary()
        if not self._allow_python_workflow or in_tf_function():
            return self._tokenize_tf(inputs)
        else:
            return self._tokenize_python(inputs)

    @preprocessing_function
    def _detokenize_tf(self, inputs):
        self._maybe_initialized_tf()
        inputs, unbatched, rectangular = convert_to_ragged_batch(inputs)
        outputs = self._fast_word_piece.detokenize(inputs)
        if unbatched:
            outputs = tf.squeeze(outputs, 0)
        return outputs

    def _detokenize_python(self, inputs):
        inputs, batched = canonicalize_python_token_inputs(inputs)
        outputs = []
        suffix_length = len(self.suffix_indicator)
        for token_ids in inputs:
            words = []
            for i, token_id in enumerate(token_ids):
                token = self.id_to_token(token_id)
                # Like tf-text, join suffix tokens to the previous word, but
                # keep a leading suffix token (and a bare suffix indicator)
                # as is.
                if (
                    i > 0
                    and len(token) > suffix_length
                    and token.startswith(self.suffix_indicator)
                ):
                    words[-1] += token[suffix_length:]
                else:
                    words.append(token)
            outputs.append(" ".join(words))
        if not batched:
            outputs = outputs[0]
        return outputs

    def detokenize(self, inputs):
        self._check_vocabulary()
        if not self._allow_python_workflow or in_tf_function():
            return self._detokenize_tf(inputs)
        else:
            return self._detokenize_python(inputs)

    def call(self, inputs, *args, training=None, **kwargs):
        return self.tokenize(inputs, *args, **kwargs)

    def compute_output_spec(self, input_spec):
        return keras.KerasTensor(
            input_spec.shape + (self.sequence_length,), dtype=self.compute_dtype
        )
