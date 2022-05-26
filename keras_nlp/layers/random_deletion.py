from typing import Any
from typing import Dict

import tensorflow as tf
from tensorflow import keras
import tensorflow_text as tf_text

class RandomDeletion(keras.layers.Layer):
    """Augments input by randomly deleting words

    Args:
        probability: probability of a word being chosen for deletion
        max_deletions: The maximum number of words to replace

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
    ...     max_deletions = 10,
    ...     stop_word_only = True,
    ... )
    >>> augmenter("dog in the house")
    <tf.Tensor: shape=(), dtype=string, numpy=b'dog house'>
    """

    def __init__(
        self, 
        probability, 
        max_deletions, 
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

        super().__init__(**kwargs)
        self.probability = probability
        self.max_deletions = max_deletions

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
            print(input_)
            tokens = tf.strings.split(input_)
            maxInd = tf.shape(tokens)[0].numpy()
            indices = []
            deletions = 0
            for i in range(maxInd):
                if tf.random.uniform(()) > self.probability:
                    indices.append(i)
                else:
                  deletions += 1
                  if (deletions > self.max_deletions):
                    break
            print(indices)
            tokens = tf.gather(tokens, indices)
            print(tokens)
            return tf.strings.join(tokens, separator=" ")

        if isinstance(inputs, tf.Tensor):
            return tf.map_fn(
                _map_fn,
                inputs,
            )

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "probability": self.probability,
                "max_deletions": self.max_deletions,
            }
        )
        return config