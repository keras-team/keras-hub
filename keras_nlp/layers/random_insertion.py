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


class RandomInsertion(keras.layers.Layer):
    """Augments input by randomly swapping words.

    The layer works by splitting the words using `tf.strings.split` computes
    then repeats the following n times:
        - Chooses 2 random indices from the input.
        - Swaps the words present at those indices.
    These 2 randomly sampled indices can also potentially be the same index.

    Args:
        swaps: Number of swaps to perform.

    Examples:

    Basic usage.
    >>> tf.random.set_seed(30)
    >>> augmenter = keras_nlp.layers.RandomInsertion(
    ...     swaps = 3
    ... )
    >>> augmenter(["I like to fly kites, do you?",
    ...     "Can we go fly some kites later?"])
    <tf.Tensor: shape=(2,), dtype=string, numpy=
    array([b'fly like do to kites, I you?',
           b'Can fly go we later? kites some'], dtype=object)>

    Augment first, then batch the dataset.
    >>> tf.random.set_seed(30)
    >>> inputs = ["I like to fly kites, do you?",
    ...     "Can we go fly some kites later?"]
    >>> augmenter = keras_nlp.layers.RandomInsertion(
    ...     swaps = 3
    ... )
    >>> ds = tf.data.Dataset.from_tensor_slices(inputs)
    >>> ds = ds.map(augmenter)
    >>> ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(2))
    >>> ds.take(1).get_single_element()
    <tf.Tensor: shape=(2,), dtype=string, numpy=
    array([b'fly like do to kites, I you?',
           b'Can fly go we later? kites some'], dtype=object)>

    Batch the inputs and then augment.
    >>> tf.random.set_seed(30)
    >>> inputs = ["I like to fly kites, do you?",
    ...     "Can we go fly some kites later?"]
    >>> augmenter = keras_nlp.layers.RandomInsertion(
    ...     swaps = 2
    ... )
    >>> ds = tf.data.Dataset.from_tensor_slices(inputs)
    >>> ds = ds.batch(2).map(augmenter)
    >>> ds.take(1).get_single_element()
    <tf.Tensor: shape=(2,), dtype=string, numpy=
    array([b'fly like I to kites, do you?',
           b'Can fly go we later? kites some'], dtype=object)>
    """

    def __init__(self, rate, max_insertions, insertion_list,
                insertion_function, name=None, **kwargs):
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
        self.insertion_function = insertion_function

    @tf.function
    def call(self, inputs):
        """Augments input by randomly swapping words.
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

        row_splits = inputs.row_splits
        positions_flat = tf.range(tf.size(inputs.flat_values))
        positions = inputs.with_flat_values(positions_flat)

        def _insert(inputs, row_splits, max_insertions,
                    insertion_list, insertion_function):
            """Inserts words into inputs at random positions.
            Args:

            Returns:

            """
            # Randomly sample a number of insertions.
            num_insertions = tf.random.uniform(
                shape=(), minval=0, maxval=max_insertions, dtype=tf.int32
            )
            # randomly choose places of insertions
            positions_to_insert = tf.random.uniform(
                shape=(num_insertions,), minval=0, maxval=tf.size(inputs),  dtype=tf.int32
            )
            # randomly choose words to insert
            words_to_insert = tf.random.uniform(
                shape=(num_insertions,), minval=0, maxval=tf.size(insertion_list), dtype=tf.int32
            )
            # insert words at random positions
            inputs = tf.tensor_scatter_nd_update(inputs, positions_to_insert,
                                                    insertion_function(words_to_insert))
            # update row_splits
            row_splits = tf.concat([row_splits[:1],
                                    tf.range(1, tf.size(inputs) + 1),
                                    row_splits[1:]], axis=0)                        
            return inputs, row_splits

        def _swap(positions):
            if tf.size(positions) == 1:
                return positions
            for _ in range(self.swaps):
                index = tf.random.uniform(
                    shape=tf.shape(positions),
                    minval=0,
                    maxval=tf.size(positions),
                    dtype=tf.int32,
                )
                index1, index2 = index[0], index[1]
                # swap items at the sampled indices with each other
                positions = tf.tensor_scatter_nd_update(
                    positions,
                    [[index1], [index2]],
                    [positions[index2], positions[index1]],
                )
            return positions

        shuffled = tf.map_fn(
            _swap,
            (positions),
            fn_output_signature=tf.RaggedTensorSpec(
                ragged_rank=positions.ragged_rank - 1, dtype=positions.dtype
            ),
        )

        shuffled.flat_values.set_shape([None])

        swapped = tf.RaggedTensor.from_row_splits(
            values=tf.gather(inputs.flat_values, shuffled.flat_values),
            row_splits=row_splits,
        )

        if scalar_input:
            swapped = tf.squeeze(swapped, 0)
        if isString:
            swapped = swapped[0]
        return swapped

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "swaps": self.swaps,
            }
        )
        return config
