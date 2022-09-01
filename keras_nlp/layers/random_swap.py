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


class RandomSwap(keras.layers.Layer):
    """Augments input by randomly swapping words.

    This layer comes in handy when you need to generate new data using swap
    augmentations as described in the paper [EDA: Easy Data Augmentation
    Techniques for Boosting Performance on Text Classification Tasks]
    (https://arxiv.org/pdf/1901.11196.pdf). The layer expects the inputs to be
    pre-split into token level inputs. This allows control over the level of
    augmentation, you can split by character for character level swaps, or by
    word for word level swaps.

    Input should be either a `tf.RaggedTensor` or a dense `tf.Tensor`, and
    either rank-1 or rank-2.

    Args:
        rate: The probability of a given token being chosen to be swapped
            with another random token.
        max_swaps: The maximum number of swaps to be performed.
        skip_list: A list of token values that should not be considered
            candidates for deletion.
        skip_fn: A function that takes as input a scalar tensor token and
            returns as output a scalar tensor True/False value. A value of
            True indicates that the token should not be considered a
            candidate for deletion. This function must be tracable--it
            should consist of tensorflow operations.
        skip_py_fn: A function that takes as input a python token value and
            returns as output `True` or `False`. A value of True
            indicates that should not be considered a candidate for deletion.
            Unlike the `skip_fn` argument, this argument need not be
            tracable--it can be any python function.
        seed: A seed for the random number generator.


    Examples:

    Word level usage.
    >>> keras.utils.set_random_seed(1337)
    >>> inputs=tf.strings.split(["Hey I like", "Keras and Tensorflow"])
    >>> augmenter=keras_nlp.layers.RandomSwap(rate=0.4, seed=42)
    >>> augmented=augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, separator=" ", axis=-1)
    <tf.Tensor: shape=(2,), dtype=string,
    numpy=array([b'like I Hey', b'and Keras Tensorflow'], dtype=object)>

    Character level usage.
    >>> keras.utils.set_random_seed(1337)
    >>> inputs=tf.strings.unicode_split(["Hey Dude", "Speed Up"], "UTF-8")
    >>> augmenter=keras_nlp.layers.RandomSwap(rate=0.4, seed=42)
    >>> augmented=augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, axis=-1)
    <tf.Tensor: shape=(2,), dtype=string,
    numpy=array([b'deD yuHe', b'SUede pp'], dtype=object)>

    Usage with skip_list.
    >>> keras.utils.set_random_seed(1337)
    >>> inputs=tf.strings.split(["Hey I like", "Keras and Tensorflow"])
    >>> augmenter=keras_nlp.layers.RandomSwap(rate=0.4,
    ...     skip_list=["Keras"], seed=42)
    >>> augmented=augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, separator=" ", axis=-1)
    <tf.Tensor: shape=(2,), dtype=string,
    numpy=array([b'like I Hey', b'Keras and Tensorflow'], dtype=object)>

    Usage with skip_fn.
    >>> def skip_fn(word):
    ...     return tf.strings.regex_full_match(word, r"[I, a].*")
    >>> keras.utils.set_random_seed(1337)
    >>> inputs=tf.strings.split(["Hey I like", "Keras and Tensorflow"])
    >>> augmenter=keras_nlp.layers.RandomSwap(rate=0.9, max_swaps=3,
    ...     skip_fn=skip_fn, seed=11)
    >>> augmented=augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, separator=" ", axis=-1)
    <tf.Tensor: shape=(2,), dtype=string,
    numpy=array([b'like I Hey', b'Keras and Tensorflow'], dtype=object)>

    Usage with skip_py_fn.
    >>> def skip_py_fn(word):
    ...     return len(word) < 4
    >>> keras.utils.set_random_seed(1337)
    >>> inputs=tf.strings.split(["He was drifting along", "With the wind"])
    >>> augmenter=keras_nlp.layers.RandomSwap(rate=0.8, max_swaps=2,
    ...     skip_py_fn=skip_py_fn, seed=15)
    >>> augmented=augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, separator=" ", axis=-1)
    <tf.Tensor: shape=(2,), dtype=string, numpy=array([b'He was along drifting',
    b'wind the With'], dtype=object)>
    """

    def __init__(
        self,
        rate,
        max_swaps=None,
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
                    "Output dtype must be one of `'string'`, `'int32'`, and "
                    f"`'int64'`. Received: dtype={dtype}"
                )

        super().__init__(name=name, **kwargs)
        self.rate = rate
        self.max_swaps = max_swaps
        self.seed = random.randint(1, 1e9) if seed is None else seed
        self._generator = tf.random.Generator.from_seed(self.seed)
        self.skip_list = skip_list
        self.skip_fn = skip_fn
        self.skip_py_fn = skip_py_fn
        if self.max_swaps is not None and self.max_swaps < 0:
            raise ValueError(
                "max_swaps must be non-negative."
                f"Received max_swaps={max_swaps}."
            )

        if [self.skip_list, self.skip_fn, self.skip_py_fn].count(None) < 2:
            raise ValueError(
                "Exactly one of skip_list, skip_fn, skip_py_fn must be "
                "provided."
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
        if self.max_swaps is not None:
            num_to_select = tf.math.minimum(num_to_select, self.max_swaps)
        num_to_select = tf.math.minimum(
            num_to_select, tf.cast(positions.row_lengths(), tf.int32)
        )
        num_to_select = tf.cast(num_to_select, tf.int64)

        def _swap(x):
            positions, inputs, num_to_select = x
            for _ in range(num_to_select):
                index = tf.random.stateless_uniform(
                    shape=[2],
                    minval=0,
                    maxval=tf.size(positions),
                    dtype=tf.int32,
                    seed=self._generator.make_seeds()[:, 0],
                )
                index1, index2 = positions[index[0]], positions[index[1]]
                # swap items at the sampled indices with each other
                inputs = tf.tensor_scatter_nd_update(
                    inputs,
                    [[index1], [index2]],
                    [inputs[index2], inputs[index1]],
                )
            return inputs

        swapped = tf.map_fn(
            _swap,
            (positions, inputs, num_to_select),
            fn_output_signature=tf.RaggedTensorSpec(
                ragged_rank=positions.ragged_rank - 1, dtype=inputs.dtype
            ),
        )
        swapped.flat_values.set_shape([None])

        if input_is_1d:
            swapped = tf.squeeze(swapped, axis=0)
        return swapped

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rate": self.rate,
                "max_swaps": self.max_swaps,
                "seed": self.seed,
                "skip_list": self.skip_list,
                "skip_fn": self.skip_fn,
                "skip_py_fn": self.skip_py_fn,
            }
        )
        return config
