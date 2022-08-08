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
from tensorflow import keras
import random

class RandomInsertion(keras.layers.Layer):
    """Augments input by randomly inserting words.

    Args:
        rate: A float in [0, 1] that is the rate of insertion
        max_replacements: An integer that is the maximum number of insertions
        insertion_list: A list of strings that are the words to insert
        insertion_fn: fn that takes in a token and returns a insertion token.
        insertion_py_fn: A python function that takes in a token and returns a
            insertion token.
        skip_list: A list of words to skip.
        skip_fn: A function that takes a word and returns True if the word
            should be skipped. This must be a traceable function of tf
            operations.
        skip_py_fn: A function that takes a word and returns True if the words
            should be skipped. Unlike skip_fn, this can be any python function
            that operates on strings, and does not need to use tf operations.
        seed: A seed for the rng.


    Examples:

    Word Level usage
    >>> keras.utils.set_random_seed(1337)
    >>> inputs=tf.strings.split(["Hey I like", "Keras and Tensorflow"])
    >>> augmenter=RandomInsertion(rate=0.4, max_insertions=2, seed=42, insertion_list=['wind'])
    >>> augmented=augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, separator=" ", axis=-1)
    <tf.Tensor: shape=(2,), dtype=string, numpy=array([b'Hey wind I like', b'Keras and wind wind Tensorflow'], dtype=object)>

    Character Level Usage
    >>> keras.utils.set_random_seed(1337)
    >>> inputs = tf.strings.unicode_split(["Hey I like", "bye bye"], "UTF-8")
    >>> augmenter = RandomInsertion(rate=0.4, max_insertions=1, seed=42, insertion_list=['y', 'z'])
    >>> augmented = augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, axis=-1)
    <tf.Tensor: shape=(2,), dtype=string, numpy=array([b'Heyy I like', b'bye bzye'], dtype=object)>

    Usage with insertion_fn
    >>> def insertion_fn(word):
    ...   if (word == "like"):
    ...     return "Car"
    ...   return "Bike"
    >>> keras.utils.set_random_seed(1337)
    >>> inputs=tf.strings.split(["Hey I like", "Keras and Tensorflow"])
    >>> augmenter = RandomInsertion(rate=0.4, max_insertions=1, seed=42, insertion_fn=insertion_fn)
    >>> augmented = augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, separator=" ", axis=-1)
    <tf.Tensor: shape=(2,), dtype=string, numpy=array([b'Hey Car I like', b'Keras and Tensorflow Bike'], dtype=object)>

    Usage with insertion_py_fn
    >>> def insertion_py_fn(word):
    ...   return word[:2]
    >>> keras.utils.set_random_seed(1337)
    >>> inputs=tf.strings.split(["Hey I like", "Keras and Tensorflow"])
    >>> augmenter = RandomInsertion(rate=0.4, max_insertions=1, seed=42, insertion_py_fn=insertion_py_fn)
    >>> augmented = augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, separator=" ", axis=-1)
    <tf.Tensor: shape=(2,), dtype=string, numpy=array([b'Hey li I like', b'Keras and Tensorflow Te'], dtype=object)>

    Usage with skip_list
    >>> keras.utils.set_random_seed(1337)
    >>> inputs=tf.strings.split(["Hey I like", "Keras and Tensorflow"])
    >>> augmenter=RandomInsertion(rate=0.4, max_insertions=1, seed=42, insertion_list=['wind'], skip_list=['like'])
    >>> augmented=augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, separator=" ", axis=-1)
    <tf.Tensor: shape=(2,), dtype=string, numpy=array([b'Hey I like', b'Keras and Tensorflow wind'], dtype=object)>

    Usage with skip_fn
    >>> def skip_fn(word):
    ...     return tf.strings.regex_full_match(word, r"\\pP")
    >>> keras.utils.set_random_seed(1337)
    >>> inputs=tf.strings.split(["Hey I like", "Keras and Tensorflow"])
    >>> augmenter=RandomInsertion(rate=0.4, max_insertions=1, seed=42, insertion_list=['wind'], skip_fn=skip_fn)
    >>> augmented=augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, separator=" ", axis=-1)
    <tf.Tensor: shape=(2,), dtype=string, numpy=array([b'Hey wind I like', b'Keras and wind Tensorflow'], dtype=object)>

    Usage with skip_py_fn
    >>> def skip_py_fn(word):
    ...     return len(word) < 2
    >>> keras.utils.set_random_seed(1337)
    >>> inputs=tf.strings.split(["Hey I like", "Keras and Tensorflow"])
    >>> augmenter=RandomInsertion(rate=0.4, max_insertions=1, seed=42, insertion_list=['wind'], skip_py_fn=skip_py_fn)
    >>> augmented=augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, separator=" ", axis=-1)
    <tf.Tensor: shape=(2,), dtype=string, numpy=array([b'Hey wind I like', b'Keras and wind Tensorflow'], dtype=object)>
    """

    def __init__(
        self,
        rate,
        max_insertions,
        insertion_list=None,
        insertion_fn=None,
        insertion_py_fn=None,
        skip_list=None,
        skip_fn=None,
        skip_py_fn=None,
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

        super().__init__(name=name, **kwargs)
        self.rate = rate
        self.max_insertions = max_insertions
        self.insertion_list = insertion_list
        self.insertion_fn = insertion_fn
        self.insertion_py_fn = insertion_py_fn
        self.skip_list = skip_list
        self.skip_fn = skip_fn
        self.skip_py_fn = skip_py_fn
        self.seed = random.randint(1, 1e9) if seed is None else seed
        self._generator = tf.random.Generator.from_seed(self.seed)

        if self.rate > 1 or self.rate < 0:
            raise ValueError(
                "Rate must be between 0 and 1 (both inclusive)."
                f"Received: rate={rate}"
            )

        if [self.skip_list, self.skip_fn, self.skip_py_fn].count(None) < 2:
            raise ValueError(
                "Exactly one of skip_list, skip_fn, skip_py_fn must be "
                "provided."
            )

        if [self.insertion_list, self.insertion_fn, self.insertion_py_fn].count(None) != 2:
            raise ValueError(
                "Exactly one of insertion_list, insertion_fn, insertion_py_fn "
                "must be provided."
            )

        if self.skip_list:
            self.StaticHashTable = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.convert_to_tensor(self.skip_list),
                    tf.convert_to_tensor([True] * len(self.skip_list)),
                ),
                default_value=False,
            )

    @tf.function
    def call(self, inputs):
        if not isinstance(inputs, (tf.Tensor, tf.RaggedTensor)):
            inputs = tf.convert_to_tensor(inputs)

        input_is_1d = False
        if inputs.shape.rank < 1 or inputs.shape.rank > 2:
            raise ValueError(
                "Input must either be rank 1 or rank 2. Received input with "
                f"rank={inputs.shape.rank}"
            )
        elif inputs.shape.rank == 1:
            input_is_1d = True
            # Add a new axis at the beginning.
            inputs = tf.expand_dims(inputs, axis=0)
        if isinstance(inputs, tf.Tensor):
            # Convert to ragged tensor.
            inputs = tf.RaggedTensor.from_tensor(inputs)

        def _check_skip(token):
            if self.skip_list:
                return self.StaticHashTable.lookup(token)
            elif self.skip_fn:
                return self.skip_fn(token)
            elif self.skip_py_fn:

                def string_fn(token):
                    return self.skip_py_fn(token.numpy().decode("utf-8"))

                def int_fn(token):
                    return self.skip_py_fn(token.numpy())

                py_fn = string_fn if inputs.dtype == tf.string else int_fn

                return tf.py_function(py_fn, [token], tf.bool)
            else:
                return False

        # Figure out how many we are going to select.
        token_counts = tf.cast(inputs.row_lengths(), "float32")
        num_to_select = tf.random.stateless_binomial(
            shape=tf.shape(token_counts),
            seed=self._generator.make_seeds()[:, 0],
            counts=token_counts,
            probs=self.rate,
        )
        if self.max_insertions is not None:
            num_to_select = tf.math.minimum(num_to_select, self.max_insertions)
        num_to_select = tf.cast(num_to_select, "int64")

        def _insert(x):
            """
            Replace words randomly
            """
            inputs, num_to_select = x
            for _ in range(num_to_select):
                index = tf.random.stateless_uniform(
                    shape=tf.shape(inputs), minval=0, maxval=tf.size(inputs), 
                    dtype=tf.int32, 
                    seed=self._generator.make_seeds()[:, 0],
                )
                replacement_word = index[0]
                insertion_location = index[1]
                original_word = inputs[replacement_word]
                if _check_skip(original_word):
                    continue
                if self.insertion_fn is not None:
                    synonym = self.insertion_fn(original_word)
                elif self.insertion_list is not None:
                    synonym_index = tf.random.stateless_uniform(
                        shape=(),
                        minval=0,
                        maxval=len(self.insertion_list),
                        dtype=tf.int32,
                        seed=self._generator.make_seeds()[:, 0],
                    )
                    synonym = tf.gather(self.insertion_list, synonym_index)
                else:
                    def _preprocess_insertion_fn(word):
                        return self.insertion_py_fn(word.numpy().decode("utf-8"))
                    synonym = tf.py_function(_preprocess_insertion_fn, [original_word], tf.string)
                # Insert the synonym at the location.
                inputs = tf.concat([inputs[:insertion_location+1], [synonym], inputs[insertion_location + 1 :]], axis=0)
            return inputs

        inserted = tf.map_fn(
            _insert,
            (inputs, num_to_select),
            fn_output_signature=tf.RaggedTensorSpec(
                ragged_rank=inputs.ragged_rank - 1, dtype=inputs.dtype
            ),
        )
        inserted.flat_values.set_shape([None])

        if input_is_1d:
            inserted = tf.squeeze(inserted, axis=0)
        return inserted

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rate": self.rate,
                "max_insertions": self.max_insertions,
                "insertion_list": self.insertion_list,
                "insertion_fn": self.insertion_fn,
                "insertion_py_fn": self.insertion_py_fn,
                "skip_list": self.skip_list,
                "skip_fn": self.skip_fn,
                "skip_py_fn": self.skip_py_fn,
            }
        )
        return config
