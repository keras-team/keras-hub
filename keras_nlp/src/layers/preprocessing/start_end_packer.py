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

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "To use `keras_nlp`, please install Tensorflow: `pip install tensorflow`. "
        "The TensorFlow package is required for data preprocessing with any backend."
    )

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.layers.preprocessing.preprocessing_layer import (
    PreprocessingLayer,
)
from keras_nlp.src.utils.tensor_utils import convert_to_ragged_batch


@keras_nlp_export("keras_nlp.layers.StartEndPacker")
class StartEndPacker(PreprocessingLayer):
    """Adds start and end tokens to a sequence and pads to a fixed length.

    This layer is useful when tokenizing inputs for tasks like translation,
    where each sequence should include a start and end marker. It should
    be called after tokenization. The layer will first trim inputs to fit, then
    add start/end tokens, and finally pad, if necessary, to `sequence_length`.

    Input data should be passed as tensors, `tf.RaggedTensor`s, or lists. For
    batched input, inputs should be a list of lists or a rank two tensor. For
    unbatched inputs, each element should be a list or a rank one tensor.

    Args:
        sequence_length: int. The desired output length.
        start_value: int/str/list/tuple. The ID(s) or token(s) that are to be
            placed at the start of each sequence. The dtype must match the dtype
            of the input tensors to the layer. If `None`, no start value will be
            added.
        end_value: int/str/list/tuple. The ID(s) or token(s) that are to be
            placed at the end of each input segment. The dtype must match the
            dtype of the input tensors to the layer. If `None`, no end value
            will be added.
        pad_value: int/str. The ID or token that is to be placed into the
            unused positions after the last segment in the sequence. If `None`,
            0 or "" will be added depending on the dtype of the input tensor.
        return_padding_mask: bool. Whether to return a boolean padding mask of
            all locations that are filled in with the `pad_value`.

    Call arguments:
        inputs: A `tf.Tensor`, `tf.RaggedTensor`, or list of python strings.
        sequence_length: Pass to override the configured `sequence_length` of
            the layer.
        add_start_value: Pass `False` to not append a start value for this
            input.
        add_end_value: Pass `False` to not append an end value for this
            input.

    Examples:

    Unbatched input (int).
    >>> inputs = [5, 6, 7]
    >>> start_end_packer = keras_nlp.layers.StartEndPacker(
    ...     sequence_length=7, start_value=1, end_value=2,
    ... )
    >>> outputs = start_end_packer(inputs)
    >>> np.array(outputs)
    array([1, 5, 6, 7, 2, 0, 0], dtype=int32)

    Batched input (int).
    >>> inputs = [[5, 6, 7], [8, 9, 10, 11, 12, 13, 14]]
    >>> start_end_packer = keras_nlp.layers.StartEndPacker(
    ...     sequence_length=6, start_value=1, end_value=2,
    ... )
    >>> outputs = start_end_packer(inputs)
    >>> np.array(outputs)
    array([[ 1,  5,  6,  7,  2,  0],
           [ 1,  8,  9, 10, 11,  2]], dtype=int32)

    Unbatched input (str).
    >>> inputs = tf.constant(["this", "is", "fun"])
    >>> start_end_packer = keras_nlp.layers.StartEndPacker(
    ...     sequence_length=6, start_value="<s>", end_value="</s>",
    ...     pad_value="<pad>"
    ... )
    >>> outputs = start_end_packer(inputs)
    >>> np.array(outputs).astype("U")
    array(['<s>', 'this', 'is', 'fun', '</s>', '<pad>'], dtype='<U5')

    Batched input (str).
    >>> inputs = tf.ragged.constant([["this", "is", "fun"], ["awesome"]])
    >>> start_end_packer = keras_nlp.layers.StartEndPacker(
    ...     sequence_length=6, start_value="<s>", end_value="</s>",
    ...     pad_value="<pad>"
    ... )
    >>> outputs = start_end_packer(inputs)
    >>> np.array(outputs).astype("U")
    array([['<s>', 'this', 'is', 'fun', '</s>', '<pad>'],
           ['<s>', 'awesome', '</s>', '<pad>', '<pad>', '<pad>']], dtype='<U7')

    Multiple start tokens.
    >>> inputs = tf.ragged.constant([["this", "is", "fun"], ["awesome"]])
    >>> start_end_packer = keras_nlp.layers.StartEndPacker(
    ...     sequence_length=6, start_value=["</s>", "<s>"], end_value="</s>",
    ...     pad_value="<pad>"
    ... )
    >>> outputs = start_end_packer(inputs)
    >>> np.array(outputs).astype("U")
    array([['</s>', '<s>', 'this', 'is', 'fun', '</s>'],
           ['</s>', '<s>', 'awesome', '</s>', '<pad>', '<pad>']], dtype='<U7')
    """

    def __init__(
        self,
        sequence_length,
        start_value=None,
        end_value=None,
        pad_value=None,
        return_padding_mask=False,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.sequence_length = sequence_length

        # Maintain private copies for config purposes.
        self._start_value = start_value
        self._end_value = end_value

        def check_special_value_type(value, value_name):
            if isinstance(value, (int, str)):
                return [value]
            if value and not isinstance(value, (list, tuple)):
                raise ValueError(
                    f"{value_name} should be of type int/str/list/tuple."
                    f"Received type: `{type(value)}`."
                )
            return value

        start_value = check_special_value_type(start_value, "start_value")
        end_value = check_special_value_type(end_value, "end_value")

        self.start_value = start_value
        self.end_value = end_value

        self.pad_value = pad_value
        self.return_padding_mask = return_padding_mask

    def call(
        self,
        inputs,
        sequence_length=None,
        add_start_value=True,
        add_end_value=True,
    ):
        inputs, unbatched, _ = convert_to_ragged_batch(inputs)

        x = inputs  # Intermediate result.

        batch_size = tf.shape(x)[0]
        sequence_length = sequence_length or self.sequence_length
        dtype = inputs.dtype

        # Concatenate start and end tokens.
        if add_start_value and self.start_value is not None:
            start_value = tf.convert_to_tensor(self.start_value, dtype=dtype)
            start_token_id_tensor = tf.repeat(
                start_value[tf.newaxis, :], repeats=batch_size, axis=0
            )
            x = tf.concat([start_token_id_tensor, x], axis=-1)
        if add_end_value and self.end_value is not None:
            end_value = tf.convert_to_tensor(self.end_value, dtype=dtype)
            end_token_id_tensor = tf.repeat(
                end_value[tf.newaxis, :], repeats=batch_size, axis=0
            )
            # Trim to leave room for end token.
            x = x[..., : sequence_length - len(self.end_value)]
            x = tf.concat([x, end_token_id_tensor], axis=-1)

        # Pad to desired length.
        outputs = x.to_tensor(
            default_value=self.pad_value,
            shape=(batch_size, sequence_length),
        )
        outputs = tf.squeeze(outputs, axis=0) if unbatched else outputs

        if self.return_padding_mask:
            mask = tf.ones_like(x, dtype="bool")
            mask = mask.to_tensor(shape=(batch_size, sequence_length))
            mask = tf.squeeze(mask, axis=0) if unbatched else mask
            return outputs, mask

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "start_value": self._start_value,
                "end_value": self._end_value,
                "pad_value": self.pad_value,
                "return_padding_mask": self.return_padding_mask,
            }
        )
        return config

    def compute_output_shape(self, inputs_shape):
        inputs_shape = list(inputs_shape)
        inputs_shape[-1] = self.sequence_length
        return tuple(inputs_shape)
