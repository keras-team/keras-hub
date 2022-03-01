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

""" Utility functions for `TransformerEncoder` and `TransformerDecoder`."""

import tensorflow as tf
from absl import logging


def compute_causal_mask(inputs):
    input_shape = tf.shape(inputs)
    batch_size, sequence_length = input_shape[0], input_shape[1]
    i = tf.range(sequence_length)[:, tf.newaxis]
    j = tf.range(sequence_length)
    mask = tf.cast(i >= j, dtype="int32")
    mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
    mult = tf.concat(
        [
            tf.expand_dims(batch_size, -1),
            tf.constant([1, 1], dtype=tf.int32),
        ],
        axis=0,
    )
    return tf.tile(mask, mult)


def merge_padding_and_attention_mask(
    inputs,
    padding_mask,
    attention_mask,
):
    """Merge padding mask with users' customized mask.

    Args:
        inputs: the input sequence.
        padding_mask: the 1D padding mask, of shape
            [batch_size, sequence_length].
        attention_mask: the 2D customized mask, of shape
            [batch_size, sequence1_length, sequence2_length].

    Return:
        A merged 2D mask or None. If only `padding_mask` is provided, the
        returned mask is padding_mask with one additional axis.
    """
    mask = padding_mask
    if hasattr(inputs, "_keras_mask"):
        if mask is None:
            # If no padding mask is explicitly provided, we look for padding
            # mask from the input data.
            mask = inputs._keras_mask
        else:
            logging.warning(
                "You are explicitly setting `padding_mask` while the `inputs` "
                "have built-in mask, so the built-in mask is ignored."
            )
    if mask is not None:
        # Add an axis for broadcasting, the attention mask should be 2D
        # (not including the batch axis).
        mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
    if attention_mask is not None:
        attention_mask = tf.cast(attention_mask, dtype=tf.int32)
        if mask is None:
            return attention_mask
        else:
            return tf.minimum(
                mask[:, tf.newaxis, :],
                attention_mask,
            )
    return mask
