import keras
import numpy as np

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.preprocessing_layer import (
    PreprocessingLayer,
)


@keras_hub_export("keras_hub.layers.v2.StartEndPacker")
class StartEndPacker(PreprocessingLayer):
    """Adds start and end tokens to a sequence and pads to a fixed length.

    This layer is useful when tokenizing inputs for tasks like translation,
    where each sequence should include a start and end marker. It should
    be called after tokenization. The layer will first trim inputs to fit, then
    add start/end tokens, and finally pad, if necessary, to `sequence_length`.

    Input data should be passed as lists. For batched input, inputs should be a
    list of lists. For unbatched inputs, each element should be a list.

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
        padding_side: str. Whether to pad the input on the "left" or "right".
            Defaults to "right".

    Call arguments:
        inputs: A list or a list of lists of python strings or ints.
        sequence_length: Pass to override the configured `sequence_length` of
            the layer.
        add_start_value: Pass `False` to not append a start value for this
            input.
        add_end_value: Pass `False` to not append an end value for this
            input.

    Examples:

    Unbatched input (int).
    >>> inputs = [5, 6, 7]
    >>> start_end_packer = keras_hub.layers.StartEndPacker(
    ...     sequence_length=7, start_value=1, end_value=2,
    ... )
    >>> outputs = start_end_packer(inputs)
    >>> np.array(outputs)
    array([1, 5, 6, 7, 2, 0, 0], dtype=int32)

    Batched input (int).
    >>> inputs = [[5, 6, 7], [8, 9, 10, 11, 12, 13, 14]]
    >>> start_end_packer = keras_hub.layers.StartEndPacker(
    ...     sequence_length=6, start_value=1, end_value=2,
    ... )
    >>> outputs = start_end_packer(inputs)
    >>> np.array(outputs)
    array([[ 1,  5,  6,  7,  2,  0],
           [ 1,  8,  9, 10, 11,  2]], dtype=int32)

    Unbatched input (str).
    >>> inputs = tf.constant(["this", "is", "fun"])
    >>> start_end_packer = keras_hub.layers.StartEndPacker(
    ...     sequence_length=6, start_value="<s>", end_value="</s>",
    ...     pad_value="<pad>"
    ... )
    >>> outputs = start_end_packer(inputs)
    >>> np.array(outputs).astype("U")
    array(['<s>', 'this', 'is', 'fun', '</s>', '<pad>'], dtype='<U5')

    Batched input (str).
    >>> inputs = tf.ragged.constant([["this", "is", "fun"], ["awesome"]])
    >>> start_end_packer = keras_hub.layers.StartEndPacker(
    ...     sequence_length=6, start_value="<s>", end_value="</s>",
    ...     pad_value="<pad>"
    ... )
    >>> outputs = start_end_packer(inputs)
    >>> np.array(outputs).astype("U")
    array([['<s>', 'this', 'is', 'fun', '</s>', '<pad>'],
           ['<s>', 'awesome', '</s>', '<pad>', '<pad>', '<pad>']], dtype='<U7')

    Multiple start tokens.
    >>> inputs = tf.ragged.constant([["this", "is", "fun"], ["awesome"]])
    >>> start_end_packer = keras_hub.layers.StartEndPacker(
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
        padding_side="right",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.sequence_length = sequence_length

        # Maintain private copies for config purposes.
        self._start_value = start_value
        self._end_value = end_value

        def check_special_value_type(value, value_name):
            if value is None:
                return None
            if isinstance(value, (int, str)):
                return [value]
            if value and not isinstance(value, (list, tuple)):
                raise ValueError(
                    f"{value_name} should be of type int/str/list/tuple."
                    f"Received type: `{type(value)}`."
                )
            return list(value)

        start_value = check_special_value_type(start_value, "start_value")
        end_value = check_special_value_type(end_value, "end_value")

        self.start_value = start_value
        self.end_value = end_value

        self.pad_value = pad_value if pad_value is not None else 0
        self.return_padding_mask = return_padding_mask
        self.padding_side = padding_side

    def _canonicalize_inputs(self, inputs):
        if isinstance(inputs, (tuple, list)):
            if isinstance(inputs[0], (tuple, list)):
                return inputs, True
            else:
                return [inputs], False
        else:
            raise ValueError(
                f"Input should be a list or a list of lists. Received: {inputs}"
            )

    def _canonicalize_value(self, inputs, values):
        if len(inputs[0]) > 0:
            first_element = inputs[0][0]
        else:
            first_element = keras.tree.flatten(inputs)[0]
        if isinstance(first_element, str):
            return [str(v) for v in values]
        else:
            return [int(v) for v in values]

    def _pad(self, x, pad_value, padding_side, sequence_length):
        if padding_side not in ("left", "right"):
            raise ValueError(
                "padding_side must be 'left' or 'right'. "
                f"Received: {padding_side}"
            )
        if padding_side == "right":
            x = [seq + [pad_value] * (sequence_length - len(seq)) for seq in x]
        else:
            x = [[pad_value] * (sequence_length - len(seq)) + seq for seq in x]
        return x

    def _canonicalize_outputs(self, outputs, dtype=None):
        if len(outputs[0]) > 0:
            first_element = outputs[0][0]
        else:
            first_element = keras.tree.flatten(outputs)[0]
        if not isinstance(first_element, str):
            return np.array(outputs, dtype=dtype or self.compute_dtype)
        else:
            return outputs

    def call(
        self,
        inputs,
        sequence_length=None,
        add_start_value=True,
        add_end_value=True,
    ):
        inputs, batched = self._canonicalize_inputs(inputs)
        sequence_length = sequence_length or self.sequence_length
        x = inputs

        # Truncate and normalize to list of lists.
        truncation_length = sequence_length
        if add_start_value and self.start_value is not None:
            truncation_length -= len(self.start_value)
        if add_end_value and self.end_value is not None:
            truncation_length -= len(self.end_value)
        x = [list(seq)[:truncation_length] for seq in x]

        # Concatenate start and end tokens.
        if add_start_value and self.start_value is not None:
            start_value = self._canonicalize_value(inputs, self.start_value)
            x = [start_value + seq for seq in x]
        if add_end_value and self.end_value is not None:
            end_value = self._canonicalize_value(inputs, self.end_value)
            x = [seq + end_value for seq in x]

        # Pad to desired length.
        outputs = self._pad(
            x,
            pad_value=self.pad_value,
            padding_side=self.padding_side,
            sequence_length=sequence_length,
        )
        outputs = outputs[0] if not batched else outputs
        outputs = self._canonicalize_outputs(outputs)

        if self.return_padding_mask:
            masks = keras.tree.map_structure(lambda _: True, x)
            masks = self._pad(
                masks,
                pad_value=False,
                padding_side=self.padding_side,
                sequence_length=sequence_length,
            )
            masks = masks[0] if not batched else masks
            masks = self._canonicalize_outputs(masks, dtype="bool")
            return outputs, masks
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
                "padding_side": self.padding_side,
            }
        )
        return config

    def compute_output_shape(self, inputs_shape):
        inputs_shape = list(inputs_shape)
        inputs_shape[-1] = self.sequence_length
        return tuple(inputs_shape)
