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

"""BERT token packing layer."""

import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow import keras


class BertPacker(keras.layers.Layer):
    """Packs multiple sequences into a single fixed width model input.

    This layer packs multiple input sequences into a single fixed width sequence
    containing start and end delimeters, forming an input suitable for BERT and
    BERT-like models.

    Takes as input a list or tuple of sequences with the layer will truncate and
    concatenate the sequences into a single sequence of `sequence_length`. The
    output sequence will always start with a `start_value` and contain an
    `end_value` after each sequence.

    If inputs are batched, inputs should be `tf.RaggedTensor`s with shape
    `[batch_size, None]` and will be packed and converted to a dense tensor with
    shape `[batch_size, sequence_length]`.

    If inputs are unbatched, inputs should be dense rank-1 tensors of any shape,
    and will be packed to shape `[sequence_length]`.

    Returns a python dictionary with three elements:
      - `"tokens"`: The packed token `tf.Tensor`.
      - `"padding_mask"`: A `tf.Tensor` with the same shape as `"tokens"`,
        containing 0s where inputs have been padded and 1s elsewhere.
      - `"segment_ids"`: A `tf.Tensor` with the same shape as `"tokens"`,
        containing an int id maching tokens to the index of input sequence they
        belonged to.

    Args:
        sequence_length: The desired output length.
        start_value: The id or token that is to be placed at the start of each
            sequence (called "[CLS]" for BERT). The dtype much mach the dtype of
            the input tensors to the layer.
        end_value: The id or token that is to be placed at the end of each
            input segment (called "[SEP]" for BERT). The dtype much mach the
            dtype of the input tensors to the layer.
        pad_value: The id or token that is to be placed into the unused
            unused positions after the last segment in the sequence
            (called "[PAD]" for BERT).
        truncator: The algorithm to truncate a list of batched segments to fit a
            per-example length limit. The value can be either `round_robin` or
            `waterfall`:
                - `"round_robin"`: Available space is assigned one token at a
                    time in a round-robin fashion to the inputs that still need
                    some, until the limit is reached.
                - `"waterfall"`: The allocation of the budget is done using a
                    "waterfall" algorithm that allocates quota in a
                    left-to-right manner and fills up the buckets until we run
                    out of budget. It support arbitrary number of segments.

    Examples:

    *Pack a single input for classification.*
    >>> seq1 = [1, 2, 3, 4]  # Replace with a real input.
    >>> packer = keras_nlp.layers.BertPacker(8, start_value=101, end_value=102)
    >>> packer(seq1)
    {
        "tokens": <tf.Tensor: shape=(8,), dtype=int32,
            numpy=array([101,   1,   2,   3,   4, 102,   0,   0], dtype=int32)>,
        "padding_mask": <tf.Tensor: shape=(8,), dtype=float32,
            numpy=array([1., 1., 1., 1., 1., 1., 0., 0.], dtype=float32)>,
        "segment_ids": <tf.Tensor: shape=(8,), dtype=float32,
            numpy=array([0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)>,
    }

    *Pack multiple inputs for classification.*
    >>> seq1 = [1, 2, 3, 4]  # Replace with a real input.
    >>> seq2 = [11, 12, 13, 14]  # Replace with a real input.
    >>> seq3 = [21, 12, 23, 24]  # Replace with a real input.
    >>> packer = keras_nlp.layers.BertPacker(8, start_value=101, end_value=102)
    >>> packer((seq1, seq2, seq3))
    {
        "tokens": <tf.Tensor: shape=(8,), dtype=int32,
            numpy=array([101,   1,   2, 102,  11, 102,  21,  102], dtype=int32)>,
        "padding_mask": <tf.Tensor: shape=(8,), dtype=float32,
            numpy=array([1., 1., 1., 1., 1., 1., 1., 1.], dtype=float32)>,
        "segment_ids": <tf.Tensor: shape=(8,), dtype=float32,
            numpy=array([0., 0., 0., 0., 1., 1., 2., 2.], dtype=float32)>,
    }

    Reference:
        [Devlin et al., 2018](https://arxiv.org/abs/1810.04805).
    """

    def __init__(
        self,
        sequence_length,
        start_value,
        end_value,
        pad_value=None,
        truncator="round_robin",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        if truncator not in ("round_robin", "waterfall"):
            raise ValueError(
                "Only 'round_robin' and 'waterfall' algorithms are "
                "supported. Received %s" % truncator
            )
        self.truncator = truncator
        self.start_value = start_value
        self.end_value = end_value
        self.pad_value = pad_value

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "start_value": self.start_value,
                "end_value": self.end_value,
                "pad_value": self.pad_value,
                "truncator": self.truncator,
            }
        )
        return config

    def _sanitize_inputs(self, inputs):
        """Force inputs to a list of rank 2 ragged tensors."""
        # Sanitize inputs.
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        if not inputs:
            raise ValueError("At least one input is required for packing")
        input_ranks = [x.shape.rank for x in inputs]
        if not all(0 < rank < 3 for rank in input_ranks):
            raise ValueError(
                "All inputs for packing must have rank 1 or 2. "
                f"Received input ranks: {input_ranks}"
            )
        if None in input_ranks or len(set(input_ranks)) > 1:
            raise ValueError(
                "All inputs for packing must have the same rank. "
                f"Received input ranks: {input_ranks}"
            )
        return inputs

    def _trim_inputs(self, inputs):
        """Trim inputs to desired length."""
        num_special_tokens = len(inputs) + 1
        if self.truncator == "round_robin":
            return tf_text.RoundRobinTrimmer(
                self.sequence_length - num_special_tokens
            ).trim(inputs)
        elif self.truncator == "waterfall":
            return tf_text.WaterfallTrimmer(
                self.sequence_length - num_special_tokens
            ).trim(inputs)
        else:
            raise ValueError("Unsupported truncator: %s" % self.truncator)

    def _combine_inputs(self, segments):
        """Trim inputs to desired length."""
        # Combine segments.
        dtype = segments[0].dtype
        start_value = tf.convert_to_tensor(self.start_value, dtype=dtype)
        end_value = tf.convert_to_tensor(self.end_value, dtype=dtype)

        start_column = tf.tile([start_value], [segments[0].nrows()])
        start_column = tf.expand_dims(start_column, 1)
        end_column = tf.tile([end_value], [segments[0].nrows()])
        end_column = tf.expand_dims(end_column, 1)
        ones_column = tf.ones_like(start_column, dtype=tf.int32)

        segments_to_combine = [start_column]
        segment_ids_to_combine = [ones_column * 0]
        for i, seg in enumerate(segments):
            # Combine all segments adding end tokens.
            segments_to_combine.append(seg)
            segments_to_combine.append(end_column)

            # Combine segment ids accounting for end tokens.
            segment_ids_to_combine.append(tf.ones_like(seg, dtype=tf.int32) * i)
            segment_ids_to_combine.append(ones_column * i)

        tokens = tf.concat(segments_to_combine, 1)
        segment_ids = tf.concat(segment_ids_to_combine, 1)
        return tokens, segment_ids

    def call(self, inputs):
        inputs = self._sanitize_inputs(inputs)

        # If rank 1, add a batch dim and convert to ragged.
        rank_1 = inputs[0].shape.rank == 1
        if rank_1:
            inputs = [tf.expand_dims(x, 0) for x in inputs]
            inputs = [tf.RaggedTensor.from_tensor(x) for x in inputs]

        segments = self._trim_inputs(inputs)
        tokens, segment_ids = self._combine_inputs(segments)
        padding_mask = tf.ones_like(segment_ids)

        # Pad to dense tensor output.
        shape = tf.cast([-1, self.sequence_length], "int64")
        tokens = tokens.to_tensor(shape=shape, default_value=self.pad_value)
        segment_ids = segment_ids.to_tensor(shape=shape)
        padding_mask = padding_mask.to_tensor(shape=shape)

        # Remove the batch dim if added.
        if rank_1:
            tokens = tf.squeeze(tokens, 0)
            segment_ids = tf.squeeze(segment_ids, 0)
            padding_mask = tf.squeeze(padding_mask, 0)

        return {
            "tokens": tokens,
            "padding_mask": padding_mask,
            "segment_ids": segment_ids,
        }
