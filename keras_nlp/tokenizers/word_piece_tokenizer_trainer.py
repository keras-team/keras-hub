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
"""Trainer for Word Piece Tokenizer."""

import tensorflow_text as tf_text
import tensorflow as tf
from tensorflow_text.tools.wordpiece_vocab import (
    wordpiece_tokenizer_learner_lib as learner,
)

from keras_nlp.tokenizers.word_piece_tokenizer import PUNCTUATION_REGEX
from keras_nlp.tokenizers.word_piece_tokenizer import (
    WHITESPACE_AND_PUNCTUATION_REGEX,
)


def compute_word_piece_vocabulary(
    data,
    vocabulary_size,
    vocabulary_output_file=None,
    lowercase=True,
    strip_accents=True,
    split=True,
    suffix_indicator="##",
    reserved_tokens=["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"],
):
    """A utility to train a Word Piece vocabulary.

    This function can be used to train a Word Piece vocabulary from an input
    dataset, or a list of filenames.

    Args:
        data: A tf.data.Dataset, or a list of filenames.
        vocabulary_size: The maximum size of a vocabulary to be trained.
        vocabulary_output_file: The location to write a vocabulary file.
        lowercase: If true, the input text will be first lowered before
            tokenization.
        strip_accents: If true, all accent marks will be removed from text
            before tokenization.
        split: If true, the input text would be split by whitespace and
            punctuation, while keeping the punctuation. Required when reading
            from a list of filenames.
        suffix_indicator: The characters prepended to a wordpiece to indicate
            that it is a suffix to another subword.
        reserved_tokens: A list of tokens that must be included in the vocabulary.

    Returns:
        Returns a list of vocabulary terms.
    """
    # Read data files.

    if not isinstance(data, (list, tf.data.Dataset)):
        raise ValueError(
            "The `data` argument must be either `tf.data.Dataset` or `list`. "
            f"Recieved: {type(data)}."
        )
    if isinstance(data, list):
        if not split:
            raise ValueError(
                "When learning a vocab from files, `split` must be `True`. "
                "To compute a vocabulary with custom split rules, load your "
                "data as a dataset, split it, and pass it to "
                "`compute_word_piece_vocabulary()` with split=False."
            )
        data = tf.data.TextLineDataset(data)

    def preprocess(text):
        """Takes in a dataset element and preprocesses it."""
        # Check for correct types.
        if text.dtype != tf.string:
            raise ValueError(
                "The dataset elements in `data` must have string dtype. "
                f"Recieved: {text.dtype}."
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

    words_data = data.map(preprocess)
    word_counts = learner.count_words(words_data)
    # Train tokenizer.
    vocab = learner.learn(
        word_counts,
        vocab_size=vocabulary_size,
        reserved_tokens=reserved_tokens,
        include_joiner_token=True,
        joiner=suffix_indicator,
    )

    if vocabulary_output_file is not None:
        vocab_text = "".join([line + "\n" for line in vocab])
        # Write vocab to file.
        with open(vocabulary_output_file, "w") as vocab_file:
            vocab_file.write(vocab_text)
    else:
        return vocab
