from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.preprocessing_layer import (
    PreprocessingLayer,
)
from keras_hub.src.utils.tensor_utils import convert_to_ragged_batch
from keras_hub.src.utils.tensor_utils import pad
from keras_hub.src.utils.tensor_utils import preprocessing_function

try:
    import tensorflow as tf
    import tensorflow_text as tf_text
except ImportError:
    tf = None
    tf_text = None


@keras_hub_export("keras_hub.layers.MultiSegmentPacker")
class MultiSegmentPacker(PreprocessingLayer):
    """Packs multiple sequences into a single fixed width model input.

    This layer packs multiple input sequences into a single fixed width sequence
    containing start and end delimeters, forming a dense input suitable for a
    classification task for BERT and BERT-like models.

    Takes as input a tuple of token segments. Each tuple element should contain
    the tokens for a segment, passed as tensors, `tf.RaggedTensor`s, or lists.
    For batched input, each element in the tuple of segments should be a list of
    lists or a rank two tensor. For unbatched inputs, each element should be a
    list or rank one tensor.

    The layer will process inputs as follows:
     - Truncate all input segments to fit within `sequence_length` according to
       the `truncate` strategy.
     - Concatenate all input segments, adding a single `start_value` at the
       start of the entire sequence, and multiple `end_value`s at the end of
       each segment.
     - Pad the resulting sequence to `sequence_length` using `pad_tokens`.
     - Calculate a separate tensor of "segment ids", with integer type and the
       same shape as the packed token output, where each integer index of the
       segment the token originated from. The segment id of the `start_value`
       is always 0, and the segment id of each `end_value` is the segment that
       precedes it.

    Args:
        sequence_length: int. The desired output length.
        start_value: int/str/list/tuple. The id(s) or token(s) that are to be
            placed at the start of each sequence (called "[CLS]" for BERT). The
            dtype must match the dtype of the input tensors to the layer.
        end_value: int/str/list/tuple. The id(s) or token(s) that are to be
            placed at the end of the last input segment (called "[SEP]" for
            BERT). The dtype must match the dtype of the input tensors to the
            layer.
        sep_value: int/str/list/tuple. The id(s) or token(s) that are to be
            placed at the end of every segment, except the last segment (called
            "[SEP]" for BERT). If `None`, `end_value` is used. The dtype must
            match the dtype of the input tensors to the layer.
        pad_value: int/str. The id or token that is to be placed into the unused
            positions after the last segment in the sequence
            (called "[PAD]" for BERT).
        truncate: str. The algorithm to truncate a list of batched segments to
            fit a per-example length limit. The value can be either
            `"round_robin"` or `"waterfall"`:
            - `"round_robin"`: Available space is assigned one token at a
                time in a round-robin fashion to the inputs that still need
                some, until the limit is reached.
            - `"waterfall"`: The allocation of the budget is done using a
                "waterfall" algorithm that allocates quota in a
                left-to-right manner and fills up the buckets until we run
                out of budget. It support arbitrary number of segments.
        padding_side: str. Whether to pad the input on the "left" or "right".
            Defaults to "right".

    Returns:
        A tuple with two elements. The first is the dense, packed token
        sequence. The second is an integer tensor of the same shape, containing
        the segment ids.

    Examples:

    *Pack a single input for classification.*
    >>> seq1 = [1, 2, 3, 4]
    >>> packer = keras_hub.layers.MultiSegmentPacker(
    ...     sequence_length=8, start_value=101, end_value=102
    ... )
    >>> token_ids, segment_ids = packer((seq1,))
    >>> np.array(token_ids)
    array([101, 1, 2, 3, 4, 102, 0, 0], dtype=int32)
    >>> np.array(segment_ids)
    array([0, 0, 0, 0, 0, 0, 0, 0], dtype=int32)

    *Pack multiple inputs for classification.*
    >>> seq1 = [1, 2, 3, 4]
    >>> seq2 = [11, 12, 13, 14]
    >>> packer = keras_hub.layers.MultiSegmentPacker(
    ...     sequence_length=8, start_value=101, end_value=102
    ... )
    >>> token_ids, segment_ids = packer((seq1, seq2))
    >>> np.array(token_ids)
    array([101, 1, 2, 3, 102,  11,  12, 102], dtype=int32)
    >>> np.array(segment_ids)
    array([0, 0, 0, 0, 0, 1, 1, 1], dtype=int32)

    *Pack multiple inputs for classification with different sep tokens.*
    >>> seq1 = [1, 2, 3, 4]
    >>> seq2 = [11, 12, 13, 14]
    >>> packer = keras_hub.layers.MultiSegmentPacker(
    ...     sequence_length=8,
    ...     start_value=101,
    ...     end_value=102,
    ...     sep_value=[102, 102],
    ... )
    >>> token_ids, segment_ids = packer((seq1, seq2))
    >>> np.array(token_ids)
    array([101,   1,   2, 102, 102,  11,  12, 102], dtype=int32)
    >>> np.array(segment_ids)
    array([0, 0, 0, 0, 0, 1, 1, 1], dtype=int32)

    Reference:
        [Devlin et al., 2018](https://arxiv.org/abs/1810.04805).
    """

    def __init__(
        self,
        sequence_length,
        start_value,
        end_value,
        sep_value=None,
        pad_value=None,
        truncate="round_robin",
        padding_side="right",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.sequence_length = sequence_length
        if truncate not in ("round_robin", "waterfall"):
            raise ValueError(
                "Only 'round_robin' and 'waterfall' algorithms are "
                "supported. Received %s" % truncate
            )
        self.truncate = truncate

        # Maintain private copies of start/end values for config purposes.
        self._start_value = start_value
        self._sep_value = sep_value
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
        if sep_value is None:
            sep_value = end_value
        sep_value = check_special_value_type(sep_value, "sep_value")
        end_value = check_special_value_type(end_value, "end_value")

        self.start_value = start_value
        self.sep_value = sep_value
        self.end_value = end_value

        self.pad_value = pad_value
        self.padding_side = padding_side

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "start_value": self._start_value,
                "end_value": self._end_value,
                "sep_value": self._sep_value,
                "pad_value": self.pad_value,
                "truncate": self.truncate,
                "padding_side": self.padding_side,
            }
        )
        return config

    def _sanitize_inputs(self, inputs):
        """Force inputs to a list of rank 2 ragged tensors."""
        # Sanitize inputs.
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        if not inputs:
            raise ValueError(
                "At least one input is required for packing. "
                f"Received: `inputs={inputs}`"
            )
        # convert_to_ragged_batch returns (x, unbatched, regtangular) triplets.
        triplets = [convert_to_ragged_batch(x) for x in inputs]
        x, unbatched, rectangular = list(zip(*triplets))
        if len(set(unbatched)) != 1:
            raise ValueError(
                "All inputs for packing must have the same rank. "
                f"Received: `inputs={inputs}`."
            )
        return x, unbatched[0]

    def _trim_inputs(self, inputs):
        """Trim inputs to desired length."""
        num_segments = len(inputs)
        num_special_tokens = (
            len(self.start_value)
            + (num_segments - 1) * len(self.sep_value)
            + len(self.end_value)
        )
        if self.truncate == "round_robin":
            return tf_text.RoundRobinTrimmer(
                self.sequence_length - num_special_tokens
            ).trim(inputs)
        elif self.truncate == "waterfall":
            return tf_text.WaterfallTrimmer(
                self.sequence_length - num_special_tokens
            ).trim(inputs)
        else:
            raise ValueError("Unsupported truncate: %s" % self.truncate)

    def _combine_inputs(
        self,
        segments,
        add_start_value=True,
        add_end_value=True,
    ):
        """Combine inputs with start and end values added."""
        dtype = segments[0].dtype
        batch_size = segments[0].nrows()
        start_value = tf.convert_to_tensor(self.start_value, dtype=dtype)
        sep_value = tf.convert_to_tensor(self.sep_value, dtype=dtype)
        end_value = tf.convert_to_tensor(self.end_value, dtype=dtype)

        start_columns = tf.repeat(
            start_value[tf.newaxis, :], repeats=batch_size, axis=0
        )
        sep_columns = tf.repeat(
            sep_value[tf.newaxis, :], repeats=batch_size, axis=0
        )
        end_columns = tf.repeat(
            end_value[tf.newaxis, :], repeats=batch_size, axis=0
        )
        ones_sep_columns = tf.ones_like(sep_columns, dtype="int32")
        ones_end_columns = tf.ones_like(end_columns, dtype="int32")

        segments_to_combine = []
        segment_ids_to_combine = []
        if add_start_value:
            segments_to_combine.append(start_columns)
            start_segment = tf.zeros_like(start_columns, dtype="int32")
            segment_ids_to_combine.append(start_segment)

        for i, seg in enumerate(segments):
            # Combine all segments.
            segments_to_combine.append(seg)

            # Combine segment ids.
            segment_ids_to_combine.append(tf.ones_like(seg, dtype="int32") * i)

            # Account for the sep/end tokens here.
            if i == len(segments) - 1:
                if add_end_value:
                    segments_to_combine.append(end_columns)
                    segment_ids_to_combine.append(ones_end_columns * i)
            else:
                segments_to_combine.append(sep_columns)
                segment_ids_to_combine.append(ones_sep_columns * i)

        token_ids = tf.concat(segments_to_combine, 1)
        segment_ids = tf.concat(segment_ids_to_combine, 1)
        return token_ids, segment_ids

    @preprocessing_function
    def call(
        self,
        inputs,
        sequence_length=None,
        add_start_value=True,
        add_end_value=True,
    ):
        inputs, unbatched = self._sanitize_inputs(inputs)

        segments = self._trim_inputs(inputs)
        token_ids, segment_ids = self._combine_inputs(
            segments,
            add_start_value=add_start_value,
            add_end_value=add_end_value,
        )
        # Pad to dense tensor output.
        sequence_length = sequence_length or self.sequence_length
        shape = tf.cast([-1, sequence_length], "int64")
        token_ids = pad(
            token_ids,
            shape=shape,
            padding_side=self.padding_side,
            pad_value=self.pad_value,
        )
        segment_ids = pad(
            segment_ids,
            shape=shape,
            padding_side=self.padding_side,
            pad_value=0,
        )
        # Remove the batch dim if added.
        if unbatched:
            token_ids = tf.squeeze(token_ids, 0)
            segment_ids = tf.squeeze(segment_ids, 0)

        return (token_ids, segment_ids)

    def compute_output_shape(self, inputs_shape):
        if isinstance(inputs_shape[0], tuple):
            inputs_shape = inputs_shape[0]
        inputs_shape = list(inputs_shape)
        inputs_shape[-1] = self.sequence_length
        return tuple(inputs_shape)
