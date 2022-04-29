from typing import Any
from typing import Dict

import tensorflow as tf
from tensorflow import keras

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


class RandomDeletion(keras.layers.Layer):
    """Augments input by randomly deleting words

    Args:
        probability: probability of a word being chosen for deletion
        max_replacements: The maximum number of words to replace
        stop_word_only: Only deletes stopwords. Defaults to False.

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
        stop_word_only: bool = False,
        **kwargs) -> None:
        super(RandomDeletion, self).__init__(**kwargs)
        self.probability = probability
        self.max_replacements = max_replacements
        self.stop_word_only = stop_word_only

    def call(self, inputs):
        replacementsPerformed = 0
        inputs = tf.strings.split(inputs)
        indices_retained = []
        i = 0
        while (
            i < len(inputs) and replacementsPerformed != self.max_replacements
        ):
            print(inputs[i])
            if tf.random.uniform(()) < self.probability:
                if self.stop_word_only:
                    if inputs[i].numpy().lower() in stop_words:
                        replacementsPerformed += 1
                else:
                    replacementsPerformed += 1
            else:
                indices_retained.append(i)
            i += 1
        # Track left over indices after max replacements are performed
        while i < len(inputs):
            indices_retained.append(i)
            i += 1
        inputs = tf.strings.reduce_join(
            tf.gather(inputs, indices_retained), separator=" "
        )
        return inputs

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "probability": self.probability,
                "max_replacements": self.max_replacements,
                "stop_word_only": self.stop_word_only,
            }
        )
        return config