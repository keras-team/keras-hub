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
    def __init__(self, name = None, **kwargs):
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

        # Shuffle items
        def _shuffle(positions):
            shuffled = tf.random.shuffle(positions)
            return shuffled

        shuffled = tf.map_fn(
            _shuffle,
            (positions),
            fn_output_signature=tf.RaggedTensorSpec(
                ragged_rank=positions.ragged_rank - 1, dtype=positions.dtype
            ),
        )

        shuffled.flat_values.set_shape([None])

        swapped = tf.RaggedTensor.from_row_splits(
              values=tf.gather(ragged_words.flat_values, shuffled.flat_values),
              row_splits=row_splits)
        
        swapped = tf.strings.reduce_join(
            swapped, axis=-1, separator=" "
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
            }
        )
        return config