# Copyright 2024 The KerasHub Authors
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


from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.tokenizers.word_piece_tokenizer import pretokenize

try:
    import tensorflow as tf
    from tensorflow_text.tools.wordpiece_vocab import (
        wordpiece_tokenizer_learner_lib as learner,
    )
except ImportError:
    tf = None
    learner = None


@keras_hub_export("keras_hub.tokenizers.compute_word_piece_vocabulary")
def compute_word_piece_vocabulary(
    data,
    vocabulary_size,
    vocabulary_output_file=None,
    lowercase=False,
    strip_accents=False,
    split=True,
    split_on_cjk=True,
    suffix_indicator="##",
    reserved_tokens=["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"],
):
    r"""A utility to train a WordPiece vocabulary.

    Trains a WordPiece vocabulary from an input dataset or a list of filenames.

    For custom data loading and pretokenization (`split=False`), the input
    `data` should be a `tf.data.Dataset`. If `data` is a list of filenames,
    the file format is required to be plain text files, and the text would be
    read in line by line during training.

    Args:
        data: A `tf.data.Dataset`, or a list of filenames.
        vocabulary_size: int. The maximum size of a vocabulary to be trained.
        vocabulary_output_file: str. The location to write a
            vocabulary file. defaults to `None`.
        lowercase: bool. If `True`, the input text will be
            lowercased before tokenization. Defaults to `False`.
        strip_accents: bool. If `True`, all accent marks will
            be removed from text before tokenization. Defaults to `False`.
        split: bool. If `True`, input will be split on
            whitespace and punctuation marks, and all punctuation marks will be
            kept as tokens. If `False`, input should be split ("pre-tokenized")
            before calling the tokenizer, and passed as a dense or ragged tensor
            of whole words. `split` is required to be `True` when `data` is a
            list of filenames. Defaults to `True`.
        split_on_cjk: bool. If `True`, input will be split
            on CJK characters, i.e., Chinese, Japanese, Korean and Vietnamese
            characters (https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)).
            Note that this is applicable only when `split` is `True`.
            Defaults to `True`.
        suffix_indicator: str. The characters prepended to a
            WordPiece to indicate that it is a suffix to another subword.
            E.g. `"##ing"`. Defaults to `"##"`.
        reserved_tokens: list of strings. A list of tokens that must be included in the vocabulary.

    Returns:
        Returns a list of vocabulary terms.

    Examples:

    Basic Usage (from Dataset).
    >>> inputs = tf.data.Dataset.from_tensor_slices(["bat sat pat mat rat"])
    >>> vocab = compute_word_piece_vocabulary(inputs, 13)
    >>> vocab
    ['[PAD]', '[CLS]', '[SEP]', '[UNK]', '[MASK]', 'a', 'b', 'm', 'p', 'r', 's', 't', '##at']
    >>> tokenizer = keras_hub.tokenizers.WordPieceTokenizer(vocabulary=vocab, oov_token="[UNK]")
    >>> outputs = inputs.map(tokenizer.tokenize)
    >>> for x in outputs:
    ...     print(x)
    tf.Tensor([ 6 12 10 12  8 12  7 12  9 12], shape=(10,), dtype=int32)

    Basic Usage (from filenames).
    ```python
    with open("test.txt", "w+") as f:
        f.write("bat sat pat mat rat\n")
    inputs = ["test.txt"]
    vocab = keras_hub.tokenizers.compute_word_piece_vocabulary(inputs, 13)
    ```

    Custom Split Usage (from Dataset).
    >>> def normalize_and_split(x):
    ...     "Strip punctuation and split on whitespace."
    ...     x = tf.strings.regex_replace(x, r"\p{P}", "")
    ...     return tf.strings.split(x)
    >>> inputs = tf.data.Dataset.from_tensor_slices(["bat sat: pat mat rat.\n"])
    >>> split_inputs = inputs.map(normalize_and_split)
    >>> vocab = compute_word_piece_vocabulary(
    ...     split_inputs, 13, split=False,
    ... )
    >>> vocab
    ['[PAD]', '[CLS]', '[SEP]', '[UNK]', '[MASK]', 'a', 'b', 'm', 'p', 'r', 's', 't', '##at']
    >>> tokenizer = keras_hub.tokenizers.WordPieceTokenizer(vocabulary=vocab)
    >>> inputs.map(tokenizer.tokenize)

    Custom Split Usage (from filenames).
    ```python
    def normalize_and_split(x):
        "Strip punctuation and split on whitespace."
        x = tf.strings.regex_replace(x, r"\p{P}", "")
        return tf.strings.split(x)
    with open("test.txt", "w+") as f:
        f.write("bat sat: pat mat rat.\n")
    inputs = tf.data.TextLineDataset(["test.txt"])
    split_inputs = inputs.map(normalize_and_split)
    vocab = keras_hub.tokenizers.compute_word_piece_vocabulary(
        split_inputs, 13, split=False
    )
    tokenizer = keras_hub.tokenizers.WordPieceTokenizer(vocabulary=vocab)
    inputs.map(tokenizer.tokenize)
    ```
    """
    # Read data files.
    if not isinstance(data, (list, tf.data.Dataset)):
        raise ValueError(
            "The `data` argument must be either `tf.data.Dataset` or `list`. "
            f"Received: {type(data)}."
        )
    if isinstance(data, list):
        # Processing list of file paths.
        if not split:
            raise ValueError(
                "When learning a vocab from files, `split` must be `True`. "
                "To compute a vocabulary with custom split rules, load your "
                "data as a dataset, split it, and pass it to "
                "`compute_word_piece_vocabulary()` with split=False."
            )
        path_ds = tf.data.Dataset.from_tensor_slices(data)
        # Uses map to read filepaths.
        data = path_ds.map(
            lambda path: tf.io.read_file(path),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    words_data = data.map(
        lambda text: pretokenize(
            text, lowercase, strip_accents, split, split_on_cjk
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    word_counts = learner.count_words(words_data)
    # Train tokenizer.
    vocab = learner.learn(
        word_counts,
        vocab_size=vocabulary_size,
        reserved_tokens=reserved_tokens,
        include_joiner_token=True,
        joiner=suffix_indicator,
    )
    if len(vocab) > vocabulary_size:
        vocab = vocab[:vocabulary_size]
    if vocabulary_output_file is not None:
        vocab_text = "".join([line + "\n" for line in vocab])
        # Write vocab to file.
        with open(vocabulary_output_file, "w", encoding="utf-8") as vocab_file:
            vocab_file.write(vocab_text)
    else:
        return vocab
