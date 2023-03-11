# Copyright 2023 The KerasNLP Authors
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


def _check_masks_shapes(inputs, padding_mask, attention_mask):
    mask = padding_mask
    if hasattr(inputs, "_keras_mask") and mask is None:
        mask = inputs._keras_mask
    if mask is not None:
        if mask._rank() != 2:
            raise ValueError(
                "`padding_mask` should have shape "
                "(batch_size, target_length). "
                f"Received shape `{mask.shape}`."
            )
    if attention_mask is not None:
        if attention_mask._rank() != 3:
            raise ValueError(
                "`attention_mask` should have shape "
                "(batch_size, target_length, source_length). "
                f"Received shape `{mask.shape}`."
            )


def compute_causal_mask(batch_size, input_length, output_length, cache_index=0):
    """Compute a causal attention mask for a transformer decoder.

    Args:
        batch_size: batch size for the mask.
        input_length: the length of key/value tensors in the attention layer.
        output_length: the length of query tensors in the attention layer.
        cache_index: the current index for cached generation. If passed, the
            query sequence will be considered to start at `cache_index` rather
            than zero. For example, a causal mask with `output_length=1` and
            `cache_index=5` would allow the query tensor to attend to the first
            five positions of the key/value tensors.

    Return:
        A causal attention mask with shape
        `(batch_size, output_length, input_length)` that can be passed to a
        attention layer.
    """
    i = tf.range(output_length)[:, tf.newaxis] + cache_index
    j = tf.range(input_length)
    mask = tf.cast(i >= j, dtype="int32")[tf.newaxis, :, :]
    return tf.broadcast_to(mask, (batch_size, output_length, input_length))


def merge_padding_and_attention_mask(
    inputs,
    padding_mask,
    attention_mask,
):
    """Merge the padding mask with a customized attention mask.

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
    _check_masks_shapes(inputs, padding_mask, attention_mask)
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
            return tf.minimum(mask, attention_mask)
    return mask
