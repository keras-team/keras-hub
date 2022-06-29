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

    def __init__(self, swaps, seed=None, name=None, **kwargs):
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
        self.seed = seed
        self._random_generator = backend.RandomGenerator(seed)

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

        def _swap(positions):
            if tf.size(positions) == 1:
                return positions
            for _ in range(self.swaps):
                index = tf.random.uniform(
                    shape=tf.shape(positions),
                    minval=0,
                    maxval=tf.size(positions),
                    dtype=tf.int32,
                    seed=self._random_generator.make_legacy_seed()
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
                "seed": self.seed,
            }
        )
        return config
