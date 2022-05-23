from typing import Any
from typing import Dict

import tensorflow as tf
from tensorflow import keras
import tensorflow_text as tf_text

stop_words = [
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "ain",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "couldn",
    "couldn't",
    "d",
    "did",
    "didn",
    "didn't",
    "do",
    "does",
    "doesn",
    "doesn't",
    "doing",
    "don",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn",
    "hadn't",
    "has",
    "hasn",
    "hasn't",
    "have",
    "haven",
    "haven't",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "isn",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "just",
    "ll",
    "m",
    "ma",
    "me",
    "mightn",
    "mightn't",
    "more",
    "most",
    "mustn",
    "mustn't",
    "my",
    "myself",
    "needn",
    "needn't",
    "no",
    "nor",
    "not",
    "now",
    "o",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "re",
    "s",
    "same",
    "shan",
    "shan't",
    "she",
    "she's",
    "should",
    "should've",
    "shouldn",
    "shouldn't",
    "so",
    "some",
    "such",
    "t",
    "than",
    "that",
    "that'll",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "ve",
    "very",
    "was",
    "wasn",
    "wasn't",
    "we",
    "were",
    "weren",
    "weren't",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "won",
    "won't",
    "wouldn",
    "wouldn't",
    "y",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
]

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

class RandomDeletion(keras.layers.Layer):
    """Augments input by randomly deleting words

    Args:
        probability: probability of a word being chosen for deletion
        max_replacements: The maximum number of words to replace
        stop_word_only: Only deletes stopwords. Defaults to False.
        split_pattern: A regex pattern to match delimiters to split. By default,
            all whitespace and punctuation marks will be split on.

    Examples:

    Basic usage.
    >>> augmenter = keras_nlp.layers.RandomDeletion(
    ...     probability = 0.3,
    ... )
    >>> augmenter(["dog dog dog dog dog"])
    <tf.Tensor: shape=(), dtype=string, numpy=b'dog dog dog dog'>

    Usage with stop_word_only.
    >>> augmenter = RandomDeletion(
    ...     probability = 0.9,
    ...     max_replacements = 10,
    ...     stop_word_only = True,
    ... )
    >>> augmenter("dog in the house")
    <tf.Tensor: shape=(), dtype=string, numpy=b'dog house'>
    """

    def __init__(
        self, 
        probability, 
        max_replacements, 
        split_pattern: str = None,
        **kwargs) -> None:
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

        if split_pattern is None:
            split_pattern = WHITESPACE_AND_PUNCTUATION_REGEX

        super().__init__(**kwargs)
        self.probability = probability
        self.max_replacements = max_replacements
        self.split_pattern = split_pattern

    def call(self, inputs):
        """Augments input by randomly deleting words

        Args:
            inputs: A tensor or nested tensor of strings to augment.

        Returns:
            A tensor or nested tensor of augmented strings.
        """
        # If input is not a tensor or ragged tensor convert it into a tensor
        if not isinstance(inputs, (tf.Tensor, tf.RaggedTensor)):
            inputs = tf.convert_to_tensor(inputs)
            inputs = tf.cast(inputs, tf.string)
        # Iterate over each innermost in tensor and either delete or keep
        # based on probability.
        def _map_fn(input_):
            # Split on whitespace and punctuation.
            tokens = tf.strings.split(input_, self.split_pattern)
            # Filter out empty strings.
            tokens = tf.boolean_mask(tokens, tf.not_equal(tokens, ""))
            # Filter out stopwords if requested.
            # if self.stop_word_only:
            #     tokens = tf.boolean_mask(tokens, tf.not_equal(tokens, STOP_WORDS))
            # Randomly select a word to delete.
            indices = tf.random.uniform(
                shape = tf.shape(tokens),
                minval = 0,
                maxval = tf.size(tokens),
                dtype = tf.int32,
            )
            # Randomly select the maximum number of replacements.
            replacements = tf.random.uniform(
                shape = [tf.minimum(tf.size(tokens), self.max_replacements)],
                minval = 0,
                maxval = tf.size(tokens),
                dtype = tf.int32,
            )
            # Delete the word at the selected index.
            tokens = tf.tensor_scatter_nd_update(tokens, indices, tf.zeros_like(indices))
            # Replace the deleted word with a random word from the same
            # vocabulary.
            tokens = tf.tensor_scatter_nd_update(
                tokens, replacements, tf.random.shuffle(tokens)
            )
            return tf.strings.reduce_join(tokens, separator = " ")

        # If input is a tensor, use map_fn to apply to each element.
        if isinstance(inputs, tf.Tensor):
            return tf.map_fn(
                _map_fn,
                inputs,
            )

        



    # def call(self, inputs):
    #     if not isinstance(inputs, (tf.Tensor, tf.RaggedTensor)):
    #         inputs = tf.convert_to_tensor(inputs)
    #     scalar_input = tf.convert_to_tensor(inputs).shape.rank == 0
    #     if scalar_input:
    #         inputs = tf.expand_dims(inputs, 0)
    #     inputs = tf_text.regex_split(
    #         inputs,
    #         delim_regex_pattern=self.split_pattern,
    #     )
        # # Randomly Delete Words in Inputs upto max replacements
        # for _ in range(self.max_replacements):
        #     inputs = tf.cond(
        #         tf.random.uniform([]) < self.probability,
        #         lambda: tf.strings.join(
        #             inputs,
        #             separator=" ",
        #         ),
        #         lambda: inputs,
        #     )
        # if scalar_input:
        #     inputs = tf.squeeze(inputs, 0)
        # return inputs
        # replacementsPerformed = 0
        # indices_retained = []
        # i = 0
        # while (
        #     i < len(inputs) and replacementsPerformed != self.max_replacements
        # ):
        #     print(inputs[i])
        #     if tf.random.uniform(()) < self.probability:
        #         if self.stop_word_only:
        #             if inputs[i].numpy().lower() in stop_words:
        #                 replacementsPerformed += 1
        #         else:
        #             replacementsPerformed += 1
        #     else:
        #         indices_retained.append(i)
        #     i += 1
        # # Track left over indices after max replacements are performed
        # while i < len(inputs):
        #     indices_retained.append(i)
        #     i += 1
        # inputs = tf.strings.reduce_join(
        #     tf.gather(inputs, indices_retained), separator=" "
        # )
        # if scalar_input:
        #     inputs = tf.squeeze(inputs, 0)
        # return inputs

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "probability": self.probability,
                "max_replacements": self.max_replacements,
                "stop_word_only": self.stop_word_only,
                "split_pattern": self.split_pattern,
            }
        )
        return config