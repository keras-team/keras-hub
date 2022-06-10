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

"""Start End Packer implementation based on `keras.layers.Layer`."""

import tensorflow as tf
from tensorflow import keras


class StartEndPacker(keras.layers.Layer):
    """Adds start and end tokens to a sequence and pads to a fixed length.

    If inputs are batched, input should be a `tf.RaggedTensor`s with shape
    `[batch_size, None]` and will be packed and converted to a dense tensor with
    shape `[batch_size, sequence_length]`.
    If inputs are unbatched, inputs should be dense rank-1 tensors of any shape,
    and will be packed to shape `[sequence_length]`.

    Args:
        sequence_length: int. The desired output length.
        start_value: int/str. The ID or token that is to be placed at the start
            of each sequence. The dtype must match the dtype of the input
            tensors to the layer. If None, no start value will be added.
        end_value: int/str. The ID or token that is to be placed at the end of
            each input segment. The dtype must match the dtype of the input
            tensors to the layer. If None, no end value will be added.
        pad_value: int/str. The ID or token that is to be placed into the
            unused positions after the last segment in the sequence. If None,
            0 or "" will be added depending on the dtype of the input tensor.
    """

    def __init__(
        self,
        sequence_length,
        start_value=None,
        end_value=None,
        pad_value=None,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.sequence_length = sequence_length
        self.start_value = start_value
        self.end_value = end_value
        self.pad_value = pad_value

    def call(self, inputs):
        input_is_tensor = isinstance(inputs, tf.Tensor)
        input_is_ragged = isinstance(inputs, tf.RaggedTensor)
        if input_is_tensor:
            if inputs.shape.rank != 1:
                raise ValueError(
                    "Input dense tensor must be of rank 1. "
                    f"Found rank={inputs.shape.rank}"
                )

            # Add a new axis at the beginning and convert to ragged tensor.
            inputs = tf.RaggedTensor.from_tensor(tf.expand_dims(inputs, axis=0))
            batch_size = 1
        elif input_is_ragged:
            if inputs.shape.rank != 2:
                raise ValueError(
                    "Input ragged tensor must be of rank 2. "
                    f"Found rank={inputs.shape.rank}"
                )
            batch_size = tf.shape(inputs)[0]
        else:
            raise ValueError(
                "Inputs must be a `tf.Tensor` or `tf.RaggedTensor`, "
                f"but got {type(inputs)}"
            )

        # Concatenate start and end tokens.
        if self.start_value is not None:
            start_token_id_tensor = tf.fill((batch_size, 1), self.start_value)
            inputs = tf.concat([start_token_id_tensor, inputs], axis=-1)
        if self.end_value is not None:
            end_token_id_tensor = tf.fill((batch_size, 1), self.end_value)
            inputs = tf.concat([inputs, end_token_id_tensor], axis=-1)

        # Pad to desired length.
        inputs = inputs.to_tensor(
            default_value=self.pad_value,
            shape=(batch_size, self.sequence_length),
        )

        if input_is_tensor:
            inputs = tf.squeeze(inputs, axis=0)

        return inputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "start_value": self.start_value,
                "end_value": self.end_value,
                "pad_value": self.pad_value,
            }
        )
        return config
