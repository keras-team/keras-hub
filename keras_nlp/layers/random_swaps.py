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


class RandomSwaps(keras.layers.Layer):
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
    >>> augmenter = keras_nlp.layers.RandomSwaps(
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
    >>> augmenter = keras_nlp.layers.RandomSwaps(
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
    >>> augmenter = keras_nlp.layers.RandomSwaps(
    ...     swaps = 2
    ... )
    >>> ds = tf.data.Dataset.from_tensor_slices(inputs)
    >>> ds = ds.batch(2).map(augmenter)
    >>> ds.take(1).get_single_element()
    <tf.Tensor: shape=(2,), dtype=string, numpy=
    array([b'fly like I to kites, do you?',
           b'Can fly go we later? kites some'], dtype=object)>
    """

    def __init__(self, swaps, name=None, **kwargs):
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

    @tf.function
    def call(self, inputs):
        """Augments input by randomly swapping words.
        Args:
            inputs: A tensor or nested tensor of strings to augment.
        Returns:
            A tensor or nested tensor of augmented strings.
        """

        def validate_and_fix_rank(inputs):
            if not isinstance(inputs, (tf.Tensor, tf.RaggedTensor)):
                inputs = tf.convert_to_tensor(inputs)
                inputs = tf.cast(inputs, tf.string)
            if inputs.shape.rank == 0 or inputs.shape.rank == 1:
                return inputs
            elif inputs.shape.rank == 2:
                if inputs.shape[1] != 1:
                    raise ValueError(
                        f"input must be of shape `[batch_size, 1]`. "
                        f"Found shape: {inputs.shape}"
                    )
                else:
                    return tf.squeeze(inputs, axis=1)
            else:
                raise ValueError(
                    f"input must be of rank 0 (scalar input), 1 or 2. "
                    f"Found rank: {inputs.shape.rank}"
                )

        isString = False
        if isinstance(inputs, str):
            inputs = [inputs]
            isString = True

        inputs = validate_and_fix_rank(inputs)

        scalar_input = inputs.shape.rank == 0
        if scalar_input:
            inputs = tf.expand_dims(inputs, 0)

        ragged_words = tf.strings.split(inputs)
        row_splits = ragged_words.row_splits
        positions_flat = tf.range(tf.size(ragged_words.flat_values))
        positions = ragged_words.with_flat_values(positions_flat)

        # Swap items
        def _swap(positions):
            # swap n times
            if tf.size(positions) == 1:
                return positions
            for _ in range(self.swaps):
                index = tf.random.uniform(
                    shape=tf.shape(positions),
                    minval=0,
                    maxval=tf.size(positions),
                    dtype=tf.int32,
                )
                # sample 2 random indices from the tensor
                shuffled = tf.random.shuffle(index)
                index1, index2 = shuffled[0], shuffled[1]
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
            values=tf.gather(ragged_words.flat_values, shuffled.flat_values),
            row_splits=row_splits,
        )

        swapped = tf.strings.reduce_join(swapped, axis=-1, separator=" ")

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
