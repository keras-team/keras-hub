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
from typing import Any
from typing import Dict

import tensorflow as tf
from tensorflow import keras


class RandomDeletion(keras.layers.Layer):
    """Augments input by randomly deleting words

    Args:
        probability: probability of a word being chosen for deletion
        max_deletions: The maximum number of words to replace

    Examples:

    Basic usage.
    >>> augmenter = keras_nlp.layers.RandomDeletion(
    ...     probability = 1,
    ...     max_deletions = 1,
    ... )
    >>> augmenter(["dog dog dog dog dog"])
    <tf.Tensor: shape=(1,), dtype=string, numpy=array([b'dog dog dog dog'],
    dtype=object)>
    """

    def __init__(self, probability, max_deletions, **kwargs) -> None:
        # Check dtype and provide a default.
        if "dtype" not in kwargs or kwargs["dtype"] is None:
            kwargs["dtype"] = tf.int32
        else:
            dtype = tf.dtypes.as_dtype(kwargs["dtype"])
            if not dtype.is_integer and dtype != tf.string:
                raise ValueError(
                    "Output dtype must be an integer type of a string. "
                    f"Received: dtype={dtype}"
                )

        super().__init__(**kwargs)
        self.probability = probability
        self.max_deletions = max_deletions

    def call(self, inputs):
        """Augments input by randomly deleting words

        Args:
            inputs: A tensor or nested tensor of strings to augment.

        Returns:
            A tensor or nested tensor of augmented strings.
        """
        # If input is not a tensor or ragged tensor convert it into a tensor
        isString = False
        if isinstance(inputs, str):
            inputs = [inputs]
            isString = True
        if not isinstance(inputs, (tf.Tensor, tf.RaggedTensor)):
            inputs = tf.convert_to_tensor(inputs)
            inputs = tf.cast(inputs, tf.string)

        def _map_fn(inputs):
            scalar_input = inputs.shape.rank == 0
            if scalar_input:
                inputs = tf.expand_dims(inputs, 0)
            ragged_words = tf.strings.split(inputs)
            row_splits = ragged_words.row_splits.numpy()
            mask = (
                tf.random.uniform(ragged_words.flat_values.shape)
                > self.probability
            )
            for i in range(len(row_splits) - 1):
                mask_range = mask[row_splits[i] : row_splits[i + 1]]
                mask_range_list = tf.unstack(mask_range)
                FalseCount = tf.reduce_sum(
                    tf.cast(tf.equal(mask_range, False), tf.int32)
                )
                if FalseCount > self.max_deletions:
                    y, idx, _ = tf.unique_with_counts(mask_range)
                    false_ind = 0 if not y[0] else 1
                    False_idxs = []
                    for j in range(len(idx)):
                        if idx[j] == false_ind:
                            False_idxs.append(j)
                    while len(False_idxs) > self.max_deletions:
                        rand_idx = random.randrange(len(False_idxs))
                        mask_range_list[False_idxs[rand_idx]] = True
                        False_idxs.pop(rand_idx)
                    mask_list = tf.unstack(mask)
                    mask_list[
                        row_splits[i] : row_splits[i + 1]
                    ] = mask_range_list
                    mask = tf.stack(mask_list)
            mask = ragged_words.with_flat_values(mask)
            deleted = tf.ragged.boolean_mask(ragged_words, mask)
            deleted = tf.strings.reduce_join(deleted, axis=-1, separator=" ")
            if scalar_input:
                deleted = tf.squeeze(deleted, 0)
            return deleted

        if isinstance(inputs, tf.Tensor):
            inputs = tf.map_fn(
                _map_fn,
                inputs,
            )
        if isString:
            inputs = inputs[0]
        return inputs

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "probability": self.probability,
                "max_deletions": self.max_deletions,
            }
        )
        return config
