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

import os
from typing import Iterable
from typing import List

import tensorflow as tf

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.tokenizers import tokenizer
from keras_nlp.utils.preset_utils import check_preset_class
from keras_nlp.utils.preset_utils import load_from_preset
from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.python_utils import format_docstring
from keras_nlp.utils.tensor_utils import assert_tf_text_installed
from keras_nlp.utils.tensor_utils import convert_to_ragged_batch
from keras_nlp.utils.tensor_utils import is_int_dtype
from keras_nlp.utils.tensor_utils import is_string_dtype

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


def pretokenize(
    text,
    lowercase=False,
    strip_accents=True,
    split=True,
    split_on_cjk=True,
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
            Note that this is applicable only when `split` is `True`. Defaults to `True`.

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
    if lowercase:
        text = tf_text.case_fold_utf8(text)
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
        text = tf_text.regex_split(
            text,
            delim_regex_pattern=split_pattern,
            keep_delim_regex_pattern=keep_split_pattern,
        )
    return text


@keras_nlp_export("keras_nlp.tokenizers.WordPieceTokenizer")
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

    References:
     - [Schuster and Nakajima, 2012](https://research.google/pubs/pub37842/)
     - [Song et al., 2020](https://arxiv.org/abs/2012.15524)

    Examples:

    Ragged outputs.
    >>> vocab = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
    >>> inputs = "The quick brown fox."
    >>> tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    ...     vocabulary=vocab,
    ...     lowercase=True,
    ... )
    >>> outputs = tokenizer(inputs)
    >>> np.array(outputs)
    array([1, 2, 3, 4, 5, 6, 7], dtype=int32)

    Dense outputs.
    >>> vocab = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
    >>> inputs = ["The quick brown fox."]
    >>> tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
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
    >>> tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    ...     vocabulary=vocab,
    ...     lowercase=True,
    ...     dtype="string",
    ... )
    >>> outputs = tokenizer(inputs)
    >>> np.array(outputs).astype("U")
    array(['the', 'qu', '##ick', 'br', '##own', 'fox', '.'], dtype='<U5')

    Detokenization.
    >>> vocab = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
    >>> inputs = "The quick brown fox."
    >>> tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    ...     vocabulary=vocab,
    ...     lowercase=True,
    ... )
    >>> outputs = tokenizer.detokenize(tokenizer.tokenize(inputs))
    >>> np.array(outputs).astype("U")
    array('the quick brown fox .', dtype='<U21')

    Custom splitting.
    >>> vocab = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
    >>> inputs = "The$quick$brown$fox"
    >>> tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    ...     vocabulary=vocab,
    ...     split=False,
    ...     lowercase=True,
    ...     dtype='string',
    ... )
    >>> split_inputs = tf.strings.split(inputs, sep="$")
    >>> outputs = tokenizer(split_inputs)
    >>> np.array(outputs).astype("U")
    array(['the', 'qu', '##ick', 'br', '##own', 'fox'], dtype='<U5')
    """

    def __init__(
        self,
        vocabulary=None,
        sequence_length: int = None,
        lowercase: bool = False,
        strip_accents: bool = False,
        split: bool = True,
        split_on_cjk: bool = True,
        suffix_indicator: str = "##",
        oov_token: str = "[UNK]",
        dtype="int32",
        **kwargs,
    ) -> None:
        assert_tf_text_installed(self.__class__.__name__)

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
        self.set_vocabulary(vocabulary)

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

    def get_vocabulary(self) -> List[str]:
        """Get the tokenizer vocabulary as a list of strings tokens."""
        self._check_vocabulary()
        return self.vocabulary

    def vocabulary_size(self) -> int:
        """Get the size of the tokenizer vocabulary."""
        self._check_vocabulary()
        return len(self.vocabulary)

    def id_to_token(self, id: int) -> str:
        """Convert an integer id to a string token."""
        self._check_vocabulary()
        if id >= self.vocabulary_size() or id < 0:
            raise ValueError(
                f"`id` must be in range [0, {self.vocabulary_size() - 1}]. "
                f"Received: {id}"
            )
        return self.vocabulary[id]

    def token_to_id(self, token: str) -> int:
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
            }
        )
        return config

    def _check_vocabulary(self):
        if self.vocabulary is None:
            raise ValueError(
                "No vocabulary has been set for WordPieceTokenizer. Make sure "
                "to pass a `vocabulary` argument when creating the layer."
            )

    def tokenize(self, inputs):
        self._check_vocabulary()
        if not isinstance(inputs, (tf.Tensor, tf.RaggedTensor)):
            inputs = tf.convert_to_tensor(inputs)

        scalar_input = inputs.shape.rank == 0
        inputs = pretokenize(
            inputs,
            self.lowercase,
            self.strip_accents,
            self.split,
            self.split_on_cjk,
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
        if scalar_input:
            tokens = tf.squeeze(tokens, 0)
            tf.ensure_shape(tokens, shape=[self.sequence_length])

        return tokens

    def detokenize(self, inputs):
        self._check_vocabulary()
        inputs, unbatched, _ = convert_to_ragged_batch(inputs)
        outputs = self._fast_word_piece.detokenize(inputs)
        if unbatched:
            outputs = tf.squeeze(outputs, 0)
        return outputs

    @classproperty
    def presets(cls):
        return {}

    @classmethod
    def from_preset(
        cls,
        preset,
        **kwargs,
    ):
        """Instantiate {{model_name}} tokenizer from preset vocabulary.

        Args:
            preset: string. Must be one of "{{preset_names}}".

        Examples:
        ```python
        # Load a preset tokenizer.
        tokenizer = {{model_name}}.from_preset("{{example_preset_name}}")

        # Tokenize some input.
        tokenizer("The quick brown fox tripped.")

        # Detokenize some input.
        tokenizer.detokenize([5, 6, 7, 8, 9])
        ```
        """
        # We support short IDs for official presets, e.g. `"bert_base_en"`.
        # Map these to a Kaggle Models handle.
        if preset in cls.presets:
            preset = cls.presets[preset]["kaggle_handle"]

        config_file = "tokenizer.json"
        check_preset_class(preset, cls, config_file=config_file)
        return load_from_preset(
            preset,
            config_file=config_file,
            config_overrides=kwargs,
        )

    def __init_subclass__(cls, **kwargs):
        # Use __init_subclass__ to setup a correct docstring for from_preset.
        super().__init_subclass__(**kwargs)

        # If the subclass does not define from_preset, assign a wrapper so that
        # each class can have a distinct docstring.
        if "from_preset" not in cls.__dict__:

            def from_preset(calling_cls, *args, **kwargs):
                return super(cls, calling_cls).from_preset(*args, **kwargs)

            cls.from_preset = classmethod(from_preset)

        # Format and assign the docstring unless the subclass has overridden it.
        if cls.from_preset.__doc__ is None:
            cls.from_preset.__func__.__doc__ = (
                WordPieceTokenizer.from_preset.__doc__
            )
            format_docstring(
                model_name=cls.__name__,
                example_preset_name=next(iter(cls.presets), ""),
                preset_names='", "'.join(cls.presets),
            )(cls.from_preset.__func__)
