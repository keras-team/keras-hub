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
import tensorflow as tf
from keras import backend
from tensorflow import keras


class RandomInsertion(keras.layers.Layer):
    """Augments input by randomly inserting words.

    Args:
        probability: A float in [0, 1] that is the probability of insertion
        max_replacements: An integer that is the maximum number of insertions
        insertion_fn: fn that takes in a token and returns a insertion token.

    Examples:

    Word Level usage
    >>> def replace_word(word):
    ...    if isinstance(word, bytes):
    ...        word = word.decode()
    ...    dict_replacement = {"like": "admire", "bye": "ciao", "Hey": "Hi"}
    ...    if (word in dict_replacement.keys()):
    ...        return dict_replacement[word]
    ...    return word
    >>> inputs = tf.strings.split(["Hey I like", "bye bye"])
    >>> augmenter = RandomInsertion(1, 5, insertion_fn = replace_word, seed = 42)
    >>> augmented = augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, separator=" ", axis=-1)
    <tf.Tensor: shape=(2,), dtype=string, numpy=array([b'Hey I admire Hi like', b'ciao bye ciao bye'], dtype=object)>

    Character Level Usage
    >>> def random_chars(word):
    ...    if isinstance(word, bytes):
    ...        word = word.decode()
    ...    if (len(word) == 0):
    ...        return "a"
    ...    return word[0]
    >>> inputs = tf.strings.unicode_split(["Hey I like", "bye bye"], "UTF-8")
    >>> augmenter = RandomInsertion(1, 5, insertion_fn = random_chars, seed = 42)
    >>> augmented = augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, axis=-1)
    <tf.Tensor: shape=(2,), dtype=string, numpy=array([b'Hey IlI like', b'byye ybye'], dtype=object)>
    """

    def __init__(
        self,
        probability,
        max_insertions,
        insertion_fn=None,
        insertion_list=None,
        seed=None,
        name=None,
        **kwargs,
    ):
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

        if insertion_fn is None and insertion_list is None:
            raise ValueError("""No insertion method provided""")

        super().__init__(name=name, **kwargs)
        self.probability = probability
        self.max_insertions = max_insertions
        self.insertion_fn = insertion_fn
        self.seed = seed
        self._random_generator = backend.RandomGenerator(seed)
        self.insertion_list = insertion_list

    @tf.function
    def call(self, inputs):
        """Augments input by randomly inserting words.
        Args:
            inputs: A tensor or nested tensor of strings to augment.
        Returns:
            A tensor or nested tensor of augmented strings.
        """
        isString = False
        if isinstance(inputs, str):
            inputs = [inputs]
            isString = True

        scalar_input = inputs.shape.rank == 0
        if scalar_input:
            inputs = tf.expand_dims(inputs, 0)

        def _insert(inputs):
            """
            Replace words randomly
            """
            # choose random number between 0 and self.max_insertions
            num_insertions = tf.random.uniform(
                shape=(),
                minval=0,
                maxval=self.max_insertions,
                dtype=tf.int32,
                seed=self._random_generator.make_legacy_seed(),
            )
            for _ in range(num_insertions):
                index = tf.random.uniform(
                    shape=tf.shape(inputs),
                    minval=0,
                    maxval=tf.size(inputs),
                    dtype=tf.int32,
                    seed=self._random_generator.make_legacy_seed(),
                )
                replacement_word = index[0]
                insertion_location = index[1]
                original_word = inputs[replacement_word]
                if self.insertion_fn is not None:
                    synonym = tf.numpy_function(
                        func=self.insertion_fn,
                        inp=[original_word],
                        Tout=tf.string,
                    )
                else:
                    synonym_index = tf.random.uniform(
                        shape=(),
                        minval=0,
                        maxval=len(self.insertion_list),
                        dtype=tf.int32,
                        seed=self._random_generator.make_legacy_seed(),
                    )
                    synonym = self.insertion_list[synonym_index]
                inputs = tf.concat(
                    [
                        inputs[:insertion_location],
                        [synonym],
                        inputs[insertion_location:],
                    ],
                    axis=0,
                )
            return inputs

        inserted = tf.map_fn(
            _insert,
            (inputs),
            fn_output_signature=tf.RaggedTensorSpec(
                ragged_rank=inputs.ragged_rank - 1, dtype=inputs.dtype
            ),
        )
        inserted.flat_values.set_shape([None])

        if scalar_input:
            inserted = tf.squeeze(inserted, 0)
        if isString:
            inserted = inserted[0]
        return inserted

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "probability": self.probability,
                "max_insertions": self.max_insertions,
                "insertion_fn": self.insertion_fn,
                "seed": self.seed,
            }
        )
        return config
