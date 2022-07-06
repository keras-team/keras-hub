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

    This layer comes in handy when you need to generate new data using deletion
    augmentation as described in the paper [EDA: Easy Data Augmentation
    Techniques for Boosting Performance on Text Classification Tasks]
    (https://arxiv.org/pdf/1901.11196.pdf). The layer expects the inputs to be
    pretokenized so that each token can be individually treated as a possible
    candidate for deletion.

    Input should be either a `tf.RaggedTensor` or a dense `tf.Tensor`, and
    either rank-1 or rank-2.

    Args:
        rate: rate of a word being chosen for deletion
        max_deletions: The maximum number of words to delete
        seed: A seed for the random number generator.

    Examples:

    Word level usage
    >>> tf.random.get_global_generator().reset_from_seed(30)
    >>> tf.random.set_seed(30)
    >>> inputs=tf.strings.split(["Hey I like", "Keras and Tensorflow"])
    >>> augmenter=keras_nlp.layers.RandomDeletion(rate=0.4, seed=42)
    >>> augmented=augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, separator=" ", axis=-1)
    <tf.Tensor: shape=(2,), dtype=string,
    numpy=array([b'Hey I', b'and Tensorflow'], dtype=object)>

    Character level usage
    >>> tf.random.get_global_generator().reset_from_seed(30)
    >>> tf.random.set_seed(30)
    >>> inputs=tf.strings.unicode_split(["Hey Dude", "Speed Up"], "UTF-8")
    >>> augmenter=keras_nlp.layers.RandomDeletion(rate=0.4, seed=42)
    >>> augmented=augmenter(inputs)
    >>> tf.strings.reduce_join(augmented, axis=-1)
    <tf.Tensor: shape=(2,), dtype=string, numpy=array([b'eDde', b'Se p'],
    dtype=object)>
    """

    def __init__(
        self, rate, max_deletions=None, seed=None, name=None, **kwargs
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
        self.max_deletions = max_deletions
        self.seed = seed
        self._random_generator = backend.RandomGenerator(seed)

        if self.rate > 1 or self.rate < 0:
            raise ValueError(
                "Rate must be between 0 and 1 (both inclusive)."
                f"Received: rate={rate}"
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
        if self.max_deletions is not None:
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

        if input_is_1d:
            inputs = tf.squeeze(inputs, axis=0)

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
