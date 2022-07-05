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


class RandomDeletion(keras.layers.Layer):
    """Augments input by randomly deleting words.

    Args:
        rate: rate of a word being chosen for deletion
        max_deletions: The maximum number of words to delete
        seed: A seed for the random number generator.

    Examples:

    Word level usage
    >>> tf.random.get_global_generator().reset_from_seed(30)
    >>> tf.random.set_seed(30)
    >>> inputs = tf.strings.split(["Hey I like", "Keras and Tensorflow"])
    >>> augmenter = keras_nlp.layers.RandomDeletion(rate = 0.4,
    ... max_deletions = 1, seed = 42)
    >>> augmented = augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, separator=" ", axis=-1)
    <tf.Tensor: shape=(2,), dtype=string,
    numpy=array([b'Hey I', b'and Tensorflow'], dtype=object)>

    Character level usage
    >>> tf.random.get_global_generator().reset_from_seed(30)
    >>> tf.random.set_seed(30)
    >>> inputs = tf.strings.unicode_split(["Hey Dude", "Speed Up"], "UTF-8")
    >>> augmenter = keras_nlp.layers.RandomDeletion(rate = 0.4,
    ... max_deletions = 1, seed = 42)
    >>> augmented = augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, axis=-1)
    <tf.Tensor: shape=(2,), dtype=string,
    numpy=array([b'Hey Dde', b'Sped Up'], dtype=object)>
    """

    def __init__(self, rate, max_deletions, seed=None, name=None, **kwargs):
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
        self.max_deletions = max_deletions
        self.seed = seed
        self._random_generator = backend.RandomGenerator(seed)

        if self.rate > 1 or self.rate < 0:
            raise ValueError(
                "Rate must be between 0 and 1 (both inclusive)."
                f"Received: rate={rate}"
            )

    def call(self, inputs):
        """Augments input by randomly deleting words.
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

        positions_flat = tf.range(tf.size(inputs.flat_values))
        positions = inputs.with_flat_values(positions_flat)

        # Figure out how many we are going to select.
        word_counts = tf.cast(inputs.row_lengths(), "float32")
        num_to_select = tf.random.stateless_binomial(
            shape=tf.shape(word_counts),
            seed=tf.random.get_global_generator().make_seeds()[:, 0],
            counts=word_counts,
            probs=self.rate,
        )
        num_to_select = tf.math.minimum(num_to_select, self.max_deletions)
        num_to_select = tf.cast(num_to_select, "int64")

        # Shuffle and trim to items that are going to be selected.
        def _shuffle_and_trim(x):
            positions, top_n = x
            shuffled = tf.random.shuffle(
                positions, seed=self._random_generator.make_legacy_seed()
            )
            return shuffled[:top_n]

        selected_for_mask = tf.map_fn(
            _shuffle_and_trim,
            (positions, num_to_select),
            fn_output_signature=tf.RaggedTensorSpec(
                ragged_rank=positions.ragged_rank - 1, dtype=positions.dtype
            ),
        )
        selected_for_mask.flat_values.set_shape([None])

        # Construct the mask which is a boolean RT
        # Scatter 0's to positions that have been selector for deletion.
        update_values = tf.zeros_like(selected_for_mask.flat_values, "int32")
        update_indices = selected_for_mask.flat_values
        update_indices = tf.expand_dims(update_indices, -1)
        update_indices = tf.cast(update_indices, "int32")
        mask_flat = tf.ones_like(inputs.flat_values, dtype="int32")
        mask_flat = tf.tensor_scatter_nd_update(
            mask_flat, update_indices, update_values
        )
        mask = tf.cast(inputs.with_flat_values(mask_flat), "bool")

        inputs = tf.ragged.boolean_mask(inputs, mask)

        if scalar_input:
            inputs = tf.squeeze(inputs, 0)
        if isString:
            inputs = inputs[0]
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rate": self.rate,
                "max_deletions": self.max_deletions,
                "seed": self.seed,
            }
        )
        return config
