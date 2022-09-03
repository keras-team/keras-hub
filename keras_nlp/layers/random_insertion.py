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
import random

import tensorflow as tf
from tensorflow import keras


class RandomInsertion(keras.layers.Layer):
    """Augments input by randomly inserting words.

    This layer comes in handy when you need to generate new data using insertion
    augmentation as described in the paper [EDA: Easy Data Augmentation
    Techniques for Boosting Performance on Text Classification Tasks]
    (https://arxiv.org/pdf/1901.11196.pdf). The layer expects the inputs to be
    pre-split into token level inputs. This allows control over the level of
    augmentation, you can split by character for character level swaps, or by
    word for word level swaps.

    Input should be either a `tf.RaggedTensor` or a dense `tf.Tensor`, and
    either rank-1 or rank-2.

    Args:
        rate: The probability of a token being chosen for insertion.
        max_insertions: The maximum number of tokens to insert.
        insertion_list: A list of tokens to use for insertion.
        insertion_fn: A function that takes a token as input and returns a
            insertion token. This must be a traceable function of tf
            operations.
        insertion_py_fn: A python function that takes in a token and returns a
            insertion token. Unlike insertion_fn, this can be any python
            function that operates on strings/integers, and does not need to use
            tf operations.
        skip_list: A list of words to skip.
        skip_fn: A function that takes a word and returns True if the word
            should be skipped. This must be a traceable function of tf
            operations.
        skip_py_fn: A function that takes a word and returns True if the words
            should be skipped. Unlike skip_fn, this can be any python function
            that operates on strings, and does not need to use tf operations.
        seed: A seed for the random number generator.


    Examples:

    Word Level usage
    >>> keras.utils.set_random_seed(1337)
    >>> inputs=tf.strings.split(["Hey I like", "Keras and Tensorflow"])
    >>> augmenter=keras_nlp.layers.RandomInsertion(rate=0.4, max_insertions=2, seed=42, insertion_list=['wind'])
    >>> augmented=augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, separator=" ", axis=-1)
    <tf.Tensor: shape=(2,), dtype=string, numpy=array([b'Hey wind I like', b'Keras and wind wind Tensorflow'], dtype=object)>

    Character Level Usage
    >>> keras.utils.set_random_seed(1337)
    >>> inputs = tf.strings.unicode_split(["Hey I like", "bye bye"], "UTF-8")
    >>> augmenter = keras_nlp.layers.RandomInsertion(rate=0.4, max_insertions=1, seed=42, insertion_list=['y', 'z'])
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
    >>> augmenter = keras_nlp.layers.RandomInsertion(rate=0.4, max_insertions=1, seed=42, insertion_fn=insertion_fn)
    >>> augmented = augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, separator=" ", axis=-1)
    <tf.Tensor: shape=(2,), dtype=string, numpy=array([b'Hey Car I like', b'Keras and Tensorflow Bike'], dtype=object)>

    Usage with insertion_py_fn
    >>> def insertion_py_fn(word):
    ...   return word[:2]
    >>> keras.utils.set_random_seed(1337)
    >>> inputs=tf.strings.split(["Hey I like", "Keras and Tensorflow"])
    >>> augmenter = keras_nlp.layers.RandomInsertion(rate=0.4, max_insertions=1, seed=42, insertion_py_fn=insertion_py_fn)
    >>> augmented = augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, separator=" ", axis=-1)
    <tf.Tensor: shape=(2,), dtype=string, numpy=array([b'Hey li I like', b'Keras and Tensorflow Te'], dtype=object)>

    Usage with skip_list
    >>> keras.utils.set_random_seed(1337)
    >>> inputs=tf.strings.split(["Hey I like", "Keras and Tensorflow"])
    >>> augmenter=keras_nlp.layers.RandomInsertion(rate=0.4, max_insertions=1, seed=42, insertion_list=['wind'], skip_list=['like'])
    >>> augmented=augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, separator=" ", axis=-1)
    <tf.Tensor: shape=(2,), dtype=string, numpy=array([b'Hey wind I like', b'Keras and wind Tensorflow'], dtype=object)>

    Usage with skip_fn
    >>> def skip_fn(word):
    ...     return tf.strings.regex_full_match(word, r"[I, a].*")
    >>> keras.utils.set_random_seed(1337)
    >>> inputs=tf.strings.split(["Hey I like", "Keras and Tensorflow"])
    >>> augmenter=keras_nlp.layers.RandomInsertion(rate=0.4, max_insertions=1, seed=42, insertion_list=['wind'], skip_fn=skip_fn)
    >>> augmented=augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, separator=" ", axis=-1)
    <tf.Tensor: shape=(2,), dtype=string, numpy=array([b'Hey wind I like', b'Keras and Tensorflow wind'], dtype=object)>

    Usage with skip_py_fn
    >>> def skip_py_fn(word):
    ...     return len(word) < 2
    >>> keras.utils.set_random_seed(1337)
    >>> inputs=tf.strings.split(["Hey I like", "Keras and Tensorflow"])
    >>> augmenter=keras_nlp.layers.RandomInsertion(rate=0.4, max_insertions=1, seed=42, insertion_list=['wind'], skip_py_fn=skip_py_fn)
    >>> augmented=augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, separator=" ", axis=-1)
    <tf.Tensor: shape=(2,), dtype=string, numpy=array([b'Hey wind I like', b'Keras and wind Tensorflow'], dtype=object)>
    """

    def __init__(
        self,
        rate,
        max_insertions=None,
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

        if self.max_insertions is not None and self.max_insertions < 0:
            raise ValueError(
                "max_insertions must be non negative"
                f"Received max_insertions={max_insertions}."
            )

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

        if [self.insertion_list, self.insertion_fn, self.insertion_py_fn].count(
            None
        ) != 2:
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

        skip_masks = None
        if self.skip_list:
            skip_masks = self.StaticHashTable.lookup(inputs.flat_values)
        elif self.skip_fn:
            skip_masks = tf.map_fn(
                self.skip_fn, inputs.flat_values, fn_output_signature=tf.bool
            )
        elif self.skip_py_fn:

            def string_fn(token):
                return self.skip_py_fn(token.numpy().decode("utf-8"))

            def int_fn(token):
                return self.skip_py_fn(token.numpy())

            py_fn = string_fn if inputs.dtype == tf.string else int_fn

            skip_masks = tf.map_fn(
                lambda x: tf.py_function(py_fn, [x], tf.bool),
                inputs.flat_values,
                fn_output_signature=tf.bool,
            )

        positions = tf.ragged.range(inputs.row_lengths())

        if skip_masks is not None:
            skip_masks = tf.logical_not(skip_masks)
            skip_masks.set_shape([None])
            positions = tf.ragged.boolean_mask(
                positions, inputs.with_flat_values(skip_masks)
            )
        # Figure out how many we are going to select.
        token_counts = tf.cast(positions.row_lengths(), "float32")
        num_to_select = tf.random.stateless_binomial(
            shape=tf.shape(token_counts),
            seed=self._generator.make_seeds()[:, 0],
            counts=token_counts,
            probs=self.rate,
        )
        if self.max_insertions is not None:
            num_to_select = tf.math.minimum(num_to_select, self.max_insertions)
        num_to_select = tf.math.minimum(
            num_to_select, tf.cast(positions.row_lengths(), tf.int32)
        )
        num_to_select = tf.cast(num_to_select, tf.int64)

        def _insert(x):
            inputs, num_to_select, positions = x
            for _ in range(num_to_select):
                tf.autograph.experimental.set_loop_options(
                    shape_invariants=[(inputs, tf.TensorShape([None]))]
                )
                index = tf.random.stateless_uniform(
                    shape=[2],
                    minval=0,
                    maxval=tf.size(positions),
                    dtype=tf.int32,
                    seed=self._generator.make_seeds()[:, 0],
                )
                insertion_prompt = positions[index[0]]
                insertion_location = positions[index[1]]
                original_word = inputs[insertion_prompt]
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

                    def string_fn(token):
                        return self.insertion_py_fn(
                            token.numpy().decode("utf-8")
                        )

                    def int_fn(token):
                        return self.insertion_py_fn(token.numpy())

                    _preprocess_insertion_fn = (
                        string_fn if inputs.dtype == tf.string else int_fn
                    )

                    synonym = tf.py_function(
                        _preprocess_insertion_fn, [original_word], inputs.dtype
                    )
                # Insert the synonym at the location.
                inputs = tf.concat(
                    [
                        inputs[: insertion_location + 1],
                        [synonym],
                        inputs[insertion_location + 1 :],
                    ],
                    axis=0,
                )
            return inputs

        inserted = tf.map_fn(
            _insert,
            (inputs, num_to_select, positions),
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
                "seed": self.seed,
            }
        )
        return config
