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


@keras.utils.register_keras_serializable(package="keras_nlp")
class StartEndPacker(keras.layers.Layer):
    """Adds start and end tokens to a sequence and pads to a fixed length.

    This layer is useful when tokenizing inputs for tasks like translation,
    where each sequence should include a start and end marker. It should
    be called after tokenization. The layer will first trim inputs to fit, then
    add start/end tokens, and finally pad, if necessary, to `sequence_length`.

    Input should be either a `tf.RaggedTensor` or a dense `tf.Tensor`, and
    either rank-1 or rank-2.

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

    Examples:

    Unbatched input (int).
    >>> input_data = tf.constant([5, 6, 7])
    >>> start_end_packer = keras_nlp.layers.StartEndPacker(
    ...     sequence_length=7, start_value=1, end_value=2,
    ... )
    >>> start_end_packer(input_data)
    <tf.Tensor: shape=(7,), dtype=int32, numpy=
    array([1, 5, 6, 7, 2, 0, 0], dtype=int32)>

    Batched input (int).
    >>> input_data = tf.ragged.constant([[5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])
    >>> start_end_packer = keras_nlp.layers.StartEndPacker(
    ...     sequence_length=6, start_value=1, end_value=2,
    ... )
    >>> start_end_packer(input_data)
    <tf.Tensor: shape=(2, 6), dtype=int32, numpy=
    array([[ 1,  5,  6,  7,  2,  0],
           [ 1,  8,  9, 10, 11,  2]], dtype=int32)>

    Unbatched input (str).
    >>> input_data = tf.constant(["this", "is", "fun"])
    >>> start_end_packer = keras_nlp.layers.StartEndPacker(
    ...     sequence_length=6, start_value="<s>", end_value="</s>",
    ...     pad_value="<pad>"
    ... )
    >>> start_end_packer(input_data)
    <tf.Tensor: shape=(6,), dtype=string, numpy=
    array([b'<s>', b'this', b'is', b'fun', b'</s>', b'<pad>'], dtype=object)>

    Batched input (str).
    >>> input_data = tf.ragged.constant([["this", "is", "fun"], ["awesome"]])
    >>> start_end_packer = keras_nlp.layers.StartEndPacker(
    ...     sequence_length=6, start_value="<s>", end_value="</s>",
    ...     pad_value="<pad>"
    ... )
    >>> start_end_packer(input_data)
    <tf.Tensor: shape=(2, 6), dtype=string, numpy=
    array([[b'<s>', b'this', b'is', b'fun', b'</s>', b'<pad>'],
           [b'<s>', b'awesome', b'</s>', b'<pad>', b'<pad>', b'<pad>']],
          dtype=object)>
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

        batch_size = tf.shape(inputs)[0]

        # Concatenate start and end tokens.
        if self.start_value is not None:
            start_token_id_tensor = tf.fill((batch_size, 1), self.start_value)
            inputs = tf.concat([start_token_id_tensor, inputs], axis=-1)
        if self.end_value is not None:
            end_token_id_tensor = tf.fill((batch_size, 1), self.end_value)

            # Trim to leave room for end token.
            inputs = inputs[..., : self.sequence_length - 1]
            inputs = tf.concat([inputs, end_token_id_tensor], axis=-1)

        # Pad to desired length.
        inputs = inputs.to_tensor(
            default_value=self.pad_value,
            shape=(batch_size, self.sequence_length),
        )

        if input_is_1d:
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
