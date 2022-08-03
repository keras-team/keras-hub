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
import random
from tensorflow.python.ops.ragged import ragged_array_ops

class RandomSwaps(keras.layers.Layer):
    """Augments input by randomly swapping words.

    The layer works by splitting the words using `tf.strings.split` computes
    then repeats the following n times:
        - Chooses 2 random indices from the input.
        - Swaps the words present at those indices.
    These 2 randomly sampled indices can also potentially be the same index.

    Args:
        swaps: Number of swaps to perform.
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
    >>> inputs = tf.strings.split(["Hey I like", "Keras and Tensorflow"])
    >>> augmenter = RandomSwaps(swaps = 3, seed = 42)
    >>> augmented = augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, separator=" ", axis=-1)
    <tf.Tensor: shape=(2,), dtype=string, numpy=array([b'I like Hey', b'and Tensorflow Keras'], dtype=object)>

    Character Level usage
    >>> inputs = tf.strings.unicode_split(["Hey I like", "bye bye"], "UTF-8")
    >>> augmenter = RandomSwaps(swaps = 1, seed = 42)
    >>> augmented = augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, axis=-1)
    <tf.Tensor: shape=(2,), dtype=string, numpy=array([b'HeI y like', b'b eybye'], dtype=object)>
    """

    def __init__(self, swaps, skip_list=None, skip_fn=None, skip_py_fn=None, 
        seed=None, name=None, **kwargs):
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
        self.swaps = swaps
        self.seed = random.randint(1, 1e9) if seed is None else seed
        self._generator = tf.random.Generator.from_seed(self.seed)
        self.skip_list = skip_list
        self.skip_fn = skip_fn
        self.skip_py_fn = skip_py_fn
        if self.swaps < 0:
            raise ValueError("Swaps must be non negative")

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

        row_splits = inputs.row_splits
        def _check_skip(token):
            if self.skip_list:
                return self.StaticHashTable.lookup(token)
            elif self.skip_fn:
                return self.skip_fn(token)
            elif self.skip_py_fn:

                def _preprocess_skip_fn(word):
                    return self.skip_py_fn(word.numpy().decode("utf-8"))

                return tf.py_function(_preprocess_skip_fn, [token], tf.bool)
            else:
                return False
        positions_flat = tf.range(tf.size(inputs.flat_values))
        positions = inputs.with_flat_values(positions_flat)

        def _swap(x):
            positions, inputs = x
            if tf.size(positions) == 1:
                return positions
            for _ in range(self.swaps):
                index = tf.random.uniform(
                    shape=tf.shape(positions),
                    minval=0,
                    maxval=tf.size(positions),
                    dtype=tf.int32,
                    seed=self.seed,
                )
                index1, index2 = index[0], index[1]
                word1 = inputs[index1]
                word2 = inputs[index2]
                if _check_skip(word1) or _check_skip(word2):
                    continue
                # swap items at the sampled indices with each other
                positions = tf.tensor_scatter_nd_update(
                    positions,
                    [[index1], [index2]],
                    [positions[index2], positions[index1]],
                )
            return positions

        shuffled = tf.map_fn(
            _swap,
            (positions, inputs),
            fn_output_signature=tf.RaggedTensorSpec(
                ragged_rank=positions.ragged_rank - 1, dtype=positions.dtype
            ),
        )

        shuffled.flat_values.set_shape([None])

        swapped = tf.RaggedTensor.from_row_splits(
            values=tf.gather(inputs.flat_values, shuffled.flat_values),
            row_splits=row_splits,
        )

        if input_is_1d:
            inputs = tf.squeeze(swapped, axis=0)
        return swapped

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "swaps": self.swaps,
                "seed": self.seed,
            }
        )
        return config
