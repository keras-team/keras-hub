import os
import re
from typing import Iterable

import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.tokenizers import tokenizer
from keras_hub.src.utils.tensor_utils import convert_to_ragged_batch
from keras_hub.src.utils.tensor_utils import is_int_dtype
from keras_hub.src.utils.tensor_utils import is_string_dtype
from keras_hub.src.utils.tensor_utils import preprocessing_function

try:
    import tensorflow as tf
    import tensorflow_text as tf_text
except ImportError:
    tf = None
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

        super().__init__(dtype=dtype, **kwargs)
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
            return

        if isinstance(vocabulary, str):
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

        self._fast_word_piece = tf_text.FastWordpieceTokenizer(
            vocab=self.vocabulary,
            token_out_type=self.compute_dtype,
            suffix_indicator=self.suffix_indicator,
            unknown_token=self.oov_token,
            no_pretokenization=True,
            support_detokenization=True,
        )
        self._update_special_token_ids()

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
        # This will be slow, but keep memory usage down compared to building a
        # . Assuming the main use case is looking up a few special tokens
        # early in the vocab, this should be fine.
        self._check_vocabulary()
        return self.vocabulary.index(token)

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

    @preprocessing_function
    def tokenize(self, inputs):
        self._check_vocabulary()
        inputs = tf.convert_to_tensor(inputs)
        unbatched = inputs.shape.rank == 0
        pattern = None
        if self.split and self.special_tokens_in_strings:
            # the idea here is to pass the special tokens regex to the
            # split function as delimiter regex pattern, so the input will
            # be splitted by them, but also the function will treat each one
            # of them as one entity that shouldn't be splitted even if they
            # have other delimiter regex pattern inside them. then pass the
            # special tokens regex also as keep delimiter regex
            # pattern, so they will not be removed.
            special_tokens = self.special_tokens
            if self._init_special_tokens:
                special_tokens += self._init_special_tokens
            pattern = get_special_tokens_pattern(special_tokens)
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

    @preprocessing_function
    def detokenize(self, inputs):
        self._check_vocabulary()
        inputs, unbatched, rectangular = convert_to_ragged_batch(inputs)
        outputs = self._fast_word_piece.detokenize(inputs)
        if unbatched:
            outputs = tf.squeeze(outputs, 0)
        return outputs

    def compute_output_spec(self, input_spec):
        return keras.KerasTensor(
            input_spec.shape + (self.sequence_length,), dtype=self.compute_dtype
        )
