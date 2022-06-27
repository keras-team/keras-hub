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
    """Augments input by randomly inserting words.

    Args:
        probability: A float in [0, 1] that is the probability of replacement
        max_replacements: An integer that is the maximum number of replacements
        replacement_list: list of candidates to uniformly sample form to replace.
            Either provide this of replacement_fn, not both.
        replacement_fn: fn that takes in a token and returns a replacement token.
        replacement_numpy_fn: python numpy version of replacement_fn.
    """

    def __init__(self, probability, max_insertions, insertion_list = None,
                insertion_fn = None, insertion_numpy_fn = None, name=None, **kwargs):
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
        self.probability = probability
        self.max_insertions = max_insertions
        self.insertion_list = insertion_list
        self.insertion_fn = insertion_fn
        self.insertion_numpy_fn = insertion_numpy_fn

    @tf.function
    def call(self, inputs):
        """Augments input by randomly replacing words.
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

        # row_splits = inputs.row_splits
        # positions_flat = tf.range(tf.size(inputs.flat_values))
        # positions = inputs.with_flat_values(positions_flat)

        def _insert(inputs):
            """
            Replace words randomly
            """
            for _ in range(self.max_insertions):
                # randomly choose an index
                index = tf.random.uniform(
                    shape=tf.shape(inputs),
                    minval=0,
                    maxval=tf.size(inputs),
                    dtype=tf.int32,
                )
                replacement_word = index[0]
                insertion_location = index[1]
                synonym = inputs[replacement_word]
                if self.insertion_numpy_fn is not None:
                    synonym = tf.numpy_function(func=self.insertion_numpy_fn, inp=[synonym], Tout=tf.string)
                    # Insert the synonym at insertion_location
                    inputs = tf.concat([inputs[:insertion_location], [synonym], inputs[insertion_location:]], axis=0)
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