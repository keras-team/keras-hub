import keras
import numpy as np

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.preprocessing_layer import (
    PreprocessingLayer,
)
from keras_hub.src.utils.tensor_utils import (
    convert_preprocessing_outputs_python,
)
from keras_hub.src.utils.tensor_utils import convert_to_list
from keras_hub.src.utils.tensor_utils import convert_to_ragged_batch
from keras_hub.src.utils.tensor_utils import in_tf_function
from keras_hub.src.utils.tensor_utils import pad
from keras_hub.src.utils.tensor_utils import preprocessing_function

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.layers.StartEndPacker")
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
        padding_side: str. Whether to pad the input on the "left" or "right".
            Defaults to "right".

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
    >>> inputs = ["this", "is", "fun"]
    >>> start_end_packer = keras_hub.layers.StartEndPacker(
    ...     sequence_length=6, start_value="<s>", end_value="</s>",
    ...     pad_value="<pad>"
    ... )
    >>> outputs = start_end_packer(inputs)
    >>> np.array(outputs).astype("U")
    array(['<s>', 'this', 'is', 'fun', '</s>', '<pad>'], dtype='<U5')

    Batched input (str).
    >>> inputs = [["this", "is", "fun"], ["awesome"]]
    >>> start_end_packer = keras_hub.layers.StartEndPacker(
    ...     sequence_length=6, start_value="<s>", end_value="</s>",
    ...     pad_value="<pad>"
    ... )
    >>> outputs = start_end_packer(inputs)
    >>> np.array(outputs).astype("U")
    array([['<s>', 'this', 'is', 'fun', '</s>', '<pad>'],
           ['<s>', 'awesome', '</s>', '<pad>', '<pad>', '<pad>']], dtype='<U7')

    Multiple start tokens.
    >>> inputs = [["this", "is", "fun"], ["awesome"]]
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
        _allow_python_workflow = kwargs.pop("_allow_python_workflow", True)
        super().__init__(
            name=name, _allow_python_workflow=_allow_python_workflow, **kwargs
        )

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
            return value

        start_value = check_special_value_type(start_value, "start_value")
        end_value = check_special_value_type(end_value, "end_value")

        self.start_value = start_value
        self.end_value = end_value

        self.pad_value = pad_value
        self.return_padding_mask = return_padding_mask
        self.padding_side = padding_side

    @preprocessing_function
    def _call_tf(
        self,
        inputs,
        sequence_length=None,
        add_start_value=True,
        add_end_value=True,
    ):
        inputs, unbatched, rectangular = convert_to_ragged_batch(inputs)
        x = inputs  # Intermediate result.

        batch_size = tf.shape(x)[0]
        sequence_length = sequence_length or self.sequence_length
        dtype = inputs.dtype
        # Truncate.
        truncation_length = sequence_length
        if add_start_value and self.start_value is not None:
            truncation_length -= len(self.start_value)
        if add_end_value and self.end_value is not None:
            truncation_length -= len(self.end_value)
        x = x[..., :truncation_length]

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
            x = tf.concat([x, end_token_id_tensor], axis=-1)

        # Pad to desired length.
        outputs = pad(
            x,
            pad_value=self.pad_value,
            padding_side=self.padding_side,
            shape=(batch_size, sequence_length),
        )
        outputs = tf.squeeze(outputs, axis=0) if unbatched else outputs

        if self.return_padding_mask:
            mask = tf.ones_like(x, dtype="bool")

            mask = pad(
                mask,
                pad_value=False,
                padding_side=self.padding_side,
                shape=(batch_size, sequence_length),
            )
            mask = tf.squeeze(mask, axis=0) if unbatched else mask
            return outputs, mask
        return outputs

    def _call_python(
        self,
        inputs,
        sequence_length=None,
        add_start_value=True,
        add_end_value=True,
    ):
        # TODO(hongyuc): Improve the performance of `_call_python`. It becomes
        # slower when encountering large inputs compared to `_call_tf`.
        def _canonicalize_inputs(inputs):
            if isinstance(inputs, (tuple, list)):
                # Fast path for common cases:
                # If the inputs are just normal python types (or lists of
                # python types), it immediately returns.
                if not inputs:
                    return [list(inputs)], False
                first = inputs[0]
                if isinstance(
                    first, (int, str, float, bool, np.integer, np.floating)
                ):
                    return [list(inputs)], False
                if isinstance(first, (tuple, list)) and (
                    not first
                    or isinstance(
                        first[0],
                        (int, str, float, bool, np.integer, np.floating),
                    )
                ):
                    return [list(x) for x in inputs], True

                # `keras.tree.map_structure` is expensive.
                inputs = keras.tree.map_structure(convert_to_list, inputs)
                if inputs and isinstance(inputs[0], (tuple, list)):
                    return inputs, True
                else:
                    return [inputs], False
            elif tf is not None and isinstance(
                inputs, (tf.Tensor, tf.RaggedTensor)
            ):
                unbatched = inputs.shape.rank == 1
                if unbatched:
                    inputs = tf.expand_dims(inputs, 0)
                if isinstance(inputs, tf.Tensor):
                    inputs = convert_to_list(inputs)
                else:
                    inputs = inputs.to_list()
                return inputs, not unbatched
            elif keras.ops.is_tensor(inputs):
                inputs = convert_to_list(inputs)
                if inputs and isinstance(inputs[0], (tuple, list)):
                    return inputs, True
                else:
                    return [inputs], False
            else:
                raise ValueError(
                    f"Input should be a list or a list of lists. "
                    f"Received: {inputs}"
                )

        def _get_type(inputs):
            for sequence in inputs:
                if sequence is not None and len(sequence) > 0:
                    return type(sequence[0])
            return int  # Default to int if all sequences are empty.

        def _canonicalize_value(values, input_type):
            if input_type is str:
                return [str(v) for v in values]
            else:
                return [int(v) for v in values]

        def _pad(x, pad_value, padding_side, sequence_length, input_type=None):
            if padding_side not in ("left", "right"):
                raise ValueError(
                    "padding_side must be 'left' or 'right'. "
                    f"Received: {padding_side}"
                )
            if pad_value is None:
                pad_value = "" if input_type is str else 0
            if padding_side == "right":
                x = [
                    seq + [pad_value] * (sequence_length - len(seq))
                    for seq in x
                ]
            else:
                x = [
                    [pad_value] * (sequence_length - len(seq)) + seq
                    for seq in x
                ]
            return x

        def _canonicalize_outputs(outputs, dtype=None):
            flat_outputs = keras.tree.flatten(outputs)
            if not flat_outputs:
                return np.array(outputs, dtype=dtype or "int32")
            first_element = flat_outputs[0]
            if not isinstance(first_element, str):
                return np.array(outputs, dtype=dtype or "int32")
            else:
                return outputs

        inputs, batched = _canonicalize_inputs(inputs)
        input_type = _get_type(inputs)
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
            start_value = _canonicalize_value(self.start_value, input_type)
            x = [start_value + seq for seq in x]
        if add_end_value and self.end_value is not None:
            end_value = _canonicalize_value(self.end_value, input_type)
            x = [seq + end_value for seq in x]

        # Pad to desired length.
        outputs = _pad(
            x,
            pad_value=self.pad_value,
            padding_side=self.padding_side,
            sequence_length=sequence_length,
            input_type=input_type,
        )
        outputs = _canonicalize_outputs(outputs)
        outputs = outputs[0] if not batched else outputs

        if self.return_padding_mask:
            masks = keras.tree.map_structure(lambda _: True, x)
            masks = _pad(
                masks,
                pad_value=False,
                padding_side=self.padding_side,
                sequence_length=sequence_length,
            )
            masks = masks[0] if not batched else masks
            masks = _canonicalize_outputs(masks, dtype="bool")
            return convert_preprocessing_outputs_python((outputs, masks))
        return convert_preprocessing_outputs_python(outputs)

    def call(
        self,
        inputs,
        sequence_length=None,
        add_start_value=True,
        add_end_value=True,
    ):
        if not self._allow_python_workflow or in_tf_function():
            return self._call_tf(
                inputs,
                sequence_length=sequence_length,
                add_start_value=add_start_value,
                add_end_value=add_end_value,
            )
        else:
            return self._call_python(
                inputs,
                sequence_length=sequence_length,
                add_start_value=add_start_value,
                add_end_value=add_end_value,
            )

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
        if self.return_padding_mask:
            return tuple(inputs_shape), tuple(inputs_shape)
        return tuple(inputs_shape)
