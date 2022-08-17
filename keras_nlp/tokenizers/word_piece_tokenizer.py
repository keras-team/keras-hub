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

from typing import Iterable
from typing import List

import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow import keras

from keras_nlp.tokenizers import tokenizer

# Pretrained Vocabularies

BASE_PATH = "https://storage.googleapis.com/keras-nlp/pretrained-tokenizers/wordpiece-tokenizer/vocabularies/"

SUPPORTED_VOCAB = {
    "en",
    "es",
    "fr",
    "ar",
    "bn",
    "hi",
    "ru",
    "id",
    "pt",
}

VOCAB_HASHES = {
    # English (en)
    "enwiki_20000_cased.txt": "dee108245d25e363d8b3a9e310148b75",
    "enwiki_20000_uncased.txt": "6a3365724775df22518fad70943cbe92",
    "enwiki_50000_cased.txt": "4a30d9a0d4fb3e2ab3fe703e7b525747",
    "enwiki_50000_uncased.txt": "8d0a7cbff5a90f7e12eca339f59c1dde",
    # French (fr)
    "frwiki_20000_cased.txt": "83edcd4ea721cc654cffe85694b53c5a",
    "frwiki_20000_uncased.txt": "d26a8be56437f0501bfd32e29cd34d3b",
    "frwiki_50000_cased.txt": "62181f510fba0551b53e86e266b2b344",
    "frwiki_50000_uncased.txt": "bb4268157ec21ff4b0a733e7ea441b3f",
    # Spanish (es)
    "eswiki_20000_cased.txt": "2e71613bdfef4d880f14d6abea805589",
    "eswiki_20000_uncased.txt": "0db01a31d4315ae1b6951ff17815cca1",
    "eswiki_50000_cased.txt": "fe87ae67e08da90f29461111d43e36d0",
    "eswiki_50000_uncased.txt": "2b86440fce72f322229ce7c374aae811",
    # Arabic (ar)
    "arwiki_20000_cased.txt": "adad214c0941f987cb3626243cfa8669",
    "arwiki_20000_uncased.txt": "7a39a909204e722ff5e6e15c26cfff09",
    "arwiki_50000_cased.txt": "1b86106ba3419174a9fc7923c68c0f31",
    "arwiki_50000_uncased.txt": "e19fd6b6054e9d7d8c18b1d63ca24050",
    # Hindi (hi)
    "hiwiki_20000_cased.txt": "6338c8bc597425c6f9dc7224c9b0bab5",
    "hiwiki_20000_uncased.txt": "1cb9ce769bae2195e3e31d0101b05e7a",
    "hiwiki_50000_cased.txt": "14ce7153c47fca47ae1b025308b1db8b",
    "hiwiki_50000_uncased.txt": "94bab6e000c8e190cc0b085aef1469fb",
    # Russian (ru)
    "ruwiki_20000_cased.txt": "64c327dd906d4f2e721c4e97cb74369e",
    "ruwiki_20000_uncased.txt": "8396c84aa19f442e0bea9a16d13eb46f",
    "ruwiki_50000_cased.txt": "01dd03cf1fcc56a5f23866501e91680d",
    "ruwiki_50000_uncased.txt": "8f10d36f15f7b0ce488d1c40a93c187d",
    # Bengali (bn)
    "bnwiki_20000_cased.txt": "ae3c5c932ce2d7aeeaa441edea66e87d",
    "bnwiki_20000_uncased.txt": "63d040dbd1fb50c13d394b33dab6a356",
    "bnwiki_50000_cased.txt": "6a7f1dd2eb58825741a33ea006e19569",
    "bnwiki_50000_uncased.txt": "8528865a47f199ce7c360e9b9f88c50f",
    # Portuguese (pt)
    "ptwiki_20000_cased.txt": "6151935d3a423020a46644e8ec7c6775",
    "ptwiki_20000_uncased.txt": "a1ba7df9e795af2ca0c5e97ce74d3ef9",
    "ptwiki_50000_cased.txt": "59226413e933ea336eb730b62b19c810",
    "ptwiki_50000_uncased.txt": "5c0bc78db111780d884e455c16d47b70",
    # Indonesian (id)
    "idwiki_20000_cased.txt": "6af5b7312fec62afcfb609bff955c3f0",
    "idwiki_20000_uncased.txt": "d0e3a893d089c9a96f12ceb8de8807e9",
    "idwiki_50000_cased.txt": "6b0d1d89dc06458f7baa4c648ccc79cb",
    "idwiki_50000_uncased.txt": "a5235e0296c0c728dda3f0eccb28240f",
}

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


def pretokenize(text, lowercase, strip_accents, split):
    """Helper function that takes in a dataset element and pretokenizes it.

    Args:
        text: `tf.Tensor` or `tf.RaggedTensor`. Input to be pretokenized.
        lowercase: bool, defaults to `True`. If true, the input text will be
            lowercased before tokenization.
        strip_accents: bool, defaults to `True`. If true, all accent marks will
            be removed from text before tokenization.
        split: bool, defaults to `True`. If true, input will be split on
            whitespace and punctuation marks, and all punctuation marks will be
            kept as tokens. If false, input should be split ("pre-tokenized")
            before calling the tokenizer, and passed as a dense or ragged tensor
            of whole words.

    Returns:
        A tensor containing the pre-processed and pre-tokenized `text`.
    """
    # Check for correct types.
    if text.dtype != tf.string:
        raise ValueError(
            "The dataset elements in `data` must have string dtype. "
            f"Received: {text.dtype}."
        )
    # Preprocess, lowercase, strip and split input data.
    if text.shape.rank == 0:
        text = tf.expand_dims(text, 0)
    if lowercase:
        text = tf_text.case_fold_utf8(text)
    if strip_accents:
        # Normalize unicode to NFD, which splits out accent mark characters.
        text = tf_text.normalize_utf8(text, "NFD")
        # Remove the accent marks.
        text = tf.strings.regex_replace(text, r"\p{Mn}", "")
    if split:
        text = tf_text.regex_split(
            text,
            delim_regex_pattern=WHITESPACE_AND_PUNCTUATION_REGEX,
            keep_delim_regex_pattern=PUNCTUATION_REGEX,
        )
    return text


def download_vocabulary(
    lang,
    vocabulary_size,
    lowercase,
    strip_accents,
    suffix_indicator,
):
    if lang not in SUPPORTED_VOCAB:
        raise ValueError(
            "This Wikipedia language code is currently not supported. Recieved "
            f"`lang={lang}`. Supported Wikipedia languages codes include "
            f"{', '.join(SUPPORTED_VOCAB)}."
        )
    if strip_accents:
        raise ValueError(
            "The pre-trained vocabularies currently does not support "
            "`strip_accents=True`."
        )
    if suffix_indicator != "##":
        raise ValueError(
            "This suffix indicator is currently not supported in pre-trained "
            'vocabularies. Use the default `suffix_indicator="##"` or '
            "provide your own vocabulary. Recieved "
            f"`suffix_indicator={suffix_indicator}`."
        )

    # Get vocabulary file
    if vocabulary_size is not None:
        # 0.95 is from the 5% buffer when training vocabularies.
        if vocabulary_size <= 20000 * 0.95:
            size = "20000"
        elif vocabulary_size <= 50000 * 0.95:
            size = "50000"
        else:
            raise ValueError(
                f"`vocabulary_size={vocabulary_size}` is not currently "
                "supported. Use a vocabulary size less than "
                f"{50000*0.95}. "
            )
    else:
        size = "50000"

    case = "uncased" if lowercase else "cased"

    filename = f"{lang}wiki_{size}_{case}.txt"
    vocabulary = keras.utils.get_file(
        filename,
        BASE_PATH + filename,
        cache_subdir="tokenizers",
        file_hash=VOCAB_HASHES[filename],
    )
    return vocabulary


class WordPieceTokenizer(tokenizer.Tokenizer):
    f"""A WordPiece tokenizer layer.

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

    Use `vocabulary` when tokenizing with your own vocabulary, and use `lang`
    when using a pre-trained vocabulary. Using both arguments is not supported.

    Pre-trained vocabularies are trained on the
    [Wikipedia dataset](https://dumps.wikimedia.org/), with each Wikipedia
    language code in `lang` corresponding to a vocabulary trained on that
    language's dataset. For example, `lang='fr'` retrives the vocabulary trained
    on the "frwiki" dataset.

    Currently supported Wikipedia language codes include:
    {", ".join(SUPPORTED_VOCAB)}.

    Args:
        vocabulary: A list of strings or a string filename path. If passing a
            list, each element of the list should be a single WordPiece token
            string. If passing a filename, the file should be a plain text file
            containing a single WordPiece token per line.
        lang: A [Wikipedia language code](https://en.wikipedia.org/wiki/List_of_Wikipedias).
            Loads the tokenizer with a vocabulary for the specified language.
        vocabulary_size: If set, the vocabulary would be truncated to
            `vocabulary_size`. Most vocabulary files are sorted in descending
            order of frequency, so the most common tokens would be kept.
        sequence_length: int. If set, the output will be converted to a dense
            tensor and padded/trimmed so all outputs are of sequence_length.
        lowercase: bool, defaults to `False`. If true, the input text will be
            lowercased before tokenization.
        strip_accents: bool, defaults to `False`. If true, all accent marks will
            be removed from text before tokenization.
        split: bool, defaults to `True`. If true, input will be split on
            whitespace and punctuation marks, and all punctuation marks will be
            kept as tokens. If false, input should be split ("pre-tokenized")
            before calling the tokenizer, and passed as a dense or ragged tensor
            of whole words.
        suffix_indicator: str, defaults to "##". The characters prepended to a
            WordPiece to indicate that it is a suffix to another subword.
            E.g. "##ing".
        oov_token: str, defaults to "[UNK]". The string value to substitute for
            an unknown token. It must be included in the vocab.

    References:
     - [Schuster and Nakajima, 2012](https://research.google/pubs/pub37842/)
     - [Song et al., 2020](https://arxiv.org/abs/2012.15524)

    Examples:

    Ragged outputs.
    >>> vocab = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
    >>> inputs = ["The quick brown fox."]
    >>> tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    ...     vocabulary=vocab,
    ...     lowercase=True,
    ... )
    >>> tokenizer(inputs)
    <tf.RaggedTensor [[1, 2, 3, 4, 5, 6, 7]]>

    Dense outputs.
    >>> vocab = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
    >>> inputs = ["The quick brown fox."]
    >>> tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    ...     vocabulary=vocab,
    ...     sequence_length=10,
    ...     lowercase=True,
    ... )
    >>> tokenizer(inputs)
    <tf.Tensor: shape=(1, 10), dtype=int32, numpy=
    array([[1, 2, 3, 4, 5, 6, 7, 0, 0, 0]], dtype=int32)>

    String output.
    >>> vocab = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
    >>> inputs = ["The quick brown fox."]
    >>> tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    ...     vocabulary=vocab,
    ...     lowercase=True,
    ...     dtype="string",
    ... )
    >>> tokenizer(inputs)
    <tf.RaggedTensor [[b'the', b'qu', b'##ick', b'br', b'##own', b'fox', b'.']]>

    Detokenization.
    >>> vocab = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
    >>> inputs = "The quick brown fox."
    >>> tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    ...     vocabulary=vocab,
    ...     lowercase=True,
    ... )
    >>> tokenizer.detokenize(tokenizer.tokenize(inputs)).numpy().decode('utf-8')
    'the quick brown fox .'

    Custom splitting.
    >>> vocab = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
    >>> inputs = ["The$quick$brown$fox"]
    >>> tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    ...     vocabulary=vocab,
    ...     split=False,
    ...     lowercase=True,
    ...     dtype='string',
    ... )
    >>> split_inputs = tf.strings.split(inputs, sep="$")
    >>> tokenizer(split_inputs)
    <tf.RaggedTensor [[b'the', b'qu', b'##ick', b'br', b'##own', b'fox']]>

    Pre-trained tokenizer.
    >>> inputs = ["The quick brown fox."]
    >>> tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    ...     lang="en",
    ...     lowercase=True,
    ...     dtype='string',
    ... )
    >>> tokenizer(inputs)
    <tf.RaggedTensor [[b'the', b'quick', b'brown', b'fox', b'.']]>
    """

    def __init__(
        self,
        vocabulary=None,
        lang: str = None,
        vocabulary_size: int = None,
        sequence_length: int = None,
        lowercase: bool = False,
        strip_accents: bool = False,
        split: bool = True,
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
                    "Output dtype must be an integer type or a string. "
                    f"Received: dtype={dtype}"
                )

        super().__init__(**kwargs)

        if vocabulary is None and lang is None:
            raise ValueError(
                "Tokenizer requires either the `vocabulary` or `lang` "
                "argument. Use `vocabulary` for custom vocabulary and `lang` "
                "for pre-trained vocabulary."
            )
        elif vocabulary is not None and lang is not None:
            raise ValueError(
                "Tokenizer requires only one of `vocabulary` or `lang` "
                "arguments. Use `vocabulary` for custom vocabulary and `lang` "
                "for pre-trained vocabulary."
            )
        elif vocabulary is None and lang is not None:
            vocabulary = download_vocabulary(
                lang,
                vocabulary_size,
                lowercase,
                strip_accents,
                suffix_indicator,
            )

        if isinstance(vocabulary, str):
            self.vocabulary = [
                line[:-1] for line in tf.io.gfile.GFile(vocabulary)
            ]
        elif isinstance(vocabulary, Iterable):
            # Make a copy.
            self.vocabulary = list(vocabulary)
        else:
            raise ValueError(
                "Vocabulary must be an file path or list of terms. "
                f"Received: vocabulary={vocabulary}"
            )
        # Truncate vocabulary.
        if vocabulary_size is not None:
            self.vocabulary = self.vocabulary[:vocabulary_size]

        if oov_token is None:
            raise ValueError("`oov_token` cannot be None.")

        self.sequence_length = sequence_length
        self.lowercase = lowercase
        self.strip_accents = strip_accents
        self.split = split
        self.suffix_indicator = suffix_indicator
        self.oov_token = oov_token

        if oov_token not in self.vocabulary:
            raise RuntimeError(
                f'Cannot find `oov_token="{self.oov_token}"` in the '
                "vocabulary.\n"
                "You can either update the vocabulary to include "
                f'`"{self.oov_token}"`, or pass a different value for '
                "the `oov_token` argument when creating the tokenizer."
            )
        self._fast_word_piece = tf_text.FastWordpieceTokenizer(
            vocab=self.vocabulary,
            token_out_type=self.compute_dtype,
            suffix_indicator=suffix_indicator,
            unknown_token=oov_token,
            no_pretokenization=True,
            support_detokenization=True,
        )

    def get_vocabulary(self) -> List[str]:
        """Get the tokenizer vocabulary as a list of strings tokens."""
        return self.vocabulary

    def vocabulary_size(self) -> int:
        """Get the size of the tokenizer vocabulary."""
        return len(self.vocabulary)

    def id_to_token(self, id: int) -> str:
        """Convert an integer id to a string token."""
        return self.vocabulary[id]

    def token_to_id(self, token: str) -> int:
        """Convert a string token to an integer id."""
        # This will be slow, but keep memory usage down compared to building a
        # . Assuming the main use case is looking up a few special tokens
        # early in the vocab, this should be fine.
        return self.vocabulary.index(token)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                # Ideally a vocabulary would be saved as a plain text asset in
                # the saved model. We have no good way to support this
                # currently, so we save the vocabulary in the config.
                "vocabulary": self.vocabulary,
                "sequence_length": self.sequence_length,
                "lowercase": self.lowercase,
                "strip_accents": self.strip_accents,
                "split": self.split,
                "suffix_indicator": self.suffix_indicator,
                "oov_token": self.oov_token,
            }
        )
        return config

    def tokenize(self, inputs):
        if not isinstance(inputs, (tf.Tensor, tf.RaggedTensor)):
            inputs = tf.convert_to_tensor(inputs)

        scalar_input = inputs.shape.rank == 0
        inputs = pretokenize(
            inputs, self.lowercase, self.strip_accents, self.split
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
        return self._fast_word_piece.detokenize(inputs)
