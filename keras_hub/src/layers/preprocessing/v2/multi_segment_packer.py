import numpy as np

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.preprocessing_layer import (
    PreprocessingLayer,
)


@keras_hub_export("keras_hub.layers.v2.MultiSegmentPacker")
class MultiSegmentPacker(PreprocessingLayer):
    """Packs multiple sequences into a single fixed width model input.

    This layer packs multiple input sequences into a single fixed width sequence
    containing start and end delimeters, forming a dense input suitable for a
    classification task for BERT and BERT-like models.

    Takes as input a tuple of token segments. Each tuple element should contain
    the tokens for a segment, passed as lists. For batched input, each element
    in the tuple of segments should be a list of lists. For unbatched inputs,
    each element should be a list.

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
        if sep_value is None:
            sep_value = end_value
        sep_value = check_special_value_type(sep_value, "sep_value")
        end_value = check_special_value_type(end_value, "end_value")

        self.start_value = start_value
        self.sep_value = sep_value
        self.end_value = end_value

        self.pad_value = pad_value if pad_value is not None else 0
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

    def _canonicalize_inputs(self, inputs):
        """Force inputs to a tuple of 2D ragged lists."""
        if isinstance(inputs, np.ndarray):
            inputs = inputs.tolist()
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        if not inputs:
            raise ValueError(
                "At least one input is required for packing. "
                f"Received: `inputs={inputs}`"
            )

        def _canonicalize_single_input(inputs):
            if isinstance(inputs, (tuple, list)):
                if isinstance(inputs[0], (tuple, list)):
                    return inputs, True
                else:
                    return [inputs], False
            else:
                raise ValueError(
                    "Input should be a list or a list of lists. "
                    f"Received: {inputs}"
                )

        # convert_to_ragged_batch returns (x, batched) triplets.
        triplets = [_canonicalize_single_input(x) for x in inputs]
        x, batched = list(zip(*triplets))
        if len(set(batched)) != 1:
            raise ValueError(
                "All inputs for packing must have the same rank. "
                f"Received: `inputs={inputs}`."
            )
        return x, batched[0]

    def _trim_inputs_round_robin(self, max_seq_length, segments):
        # segments: list of lists of lists
        num_segments = len(segments)
        batch_size = len(segments[0])
        trimmed_segments = [
            [[] for _ in range(batch_size)] for _ in range(num_segments)
        ]
        for b in range(batch_size):
            lengths = [len(segments[s][b]) for s in range(num_segments)]
            if sum(lengths) <= max_seq_length:
                for s in range(num_segments):
                    trimmed_segments[s][b] = segments[s][b]
                continue

            # Binary search for the cutoff priority
            # Priority of token k in segment s is: k * num_segments + s
            # We want to find the smallest P such that
            # count(priority <= P) >= max_seq_length
            low = -1
            high = num_segments * max(lengths)
            cutoff_priority = high
            while low <= high:
                mid = (low + high) // 2
                count = 0
                for s in range(num_segments):
                    max_index = (mid - s) // num_segments
                    if max_index >= 0:
                        count += min(lengths[s], max_index + 1)

                if count >= max_seq_length:
                    cutoff_priority = mid
                    high = mid - 1
                else:
                    low = mid + 1

            # Apply the cutoff
            for s in range(num_segments):
                max_index = (cutoff_priority - s) // num_segments
                trimmed_segments[s][b] = segments[s][b][: max_index + 1]
        return trimmed_segments

    def _trim_inputs_waterfall(self, max_seq_length, segments):
        num_segments = len(segments)
        batch_size = len(segments[0])
        trimmed_segments = [
            [[] for _ in range(batch_size)] for _ in range(num_segments)
        ]
        for b in range(batch_size):
            remaining_budget = max_seq_length

            # Check if total length is within limit
            for s in range(num_segments):
                seg_len = len(segments[s][b])
                if remaining_budget <= 0:
                    trimmed_segments[s][b] = []
                elif seg_len <= remaining_budget:
                    trimmed_segments[s][b] = segments[s][b]
                    remaining_budget -= seg_len
                else:
                    trimmed_segments[s][b] = segments[s][b][:remaining_budget]
                    remaining_budget = 0
        return trimmed_segments

    def _trim_inputs(self, inputs):
        """Trim inputs to desired length."""
        num_segments = len(inputs)
        num_special_tokens = (
            len(self.start_value)
            + (num_segments - 1) * len(self.sep_value)
            + len(self.end_value)
        )
        if self.truncate == "round_robin":
            return self._trim_inputs_round_robin(
                self.sequence_length - num_special_tokens, inputs
            )
        elif self.truncate == "waterfall":
            return self._trim_inputs_waterfall(
                self.sequence_length - num_special_tokens, inputs
            )
        else:
            raise ValueError("Unsupported truncate: %s" % self.truncate)

    def _combine_inputs(
        self,
        segments,
        add_start_value=True,
        add_end_value=True,
    ):
        """Combine inputs with start and end values added."""
        batch_size = len(segments[0])
        token_ids = []
        segment_ids = []

        for b in range(batch_size):
            example_tokens = []
            example_segment_ids = []

            if add_start_value:
                example_tokens.extend(self.start_value)
                example_segment_ids.extend([0] * len(self.start_value))

            for i in range(len(segments)):
                seg = segments[i][b]
                example_tokens.extend(seg)
                example_segment_ids.extend([i] * len(seg))

                # Account for the sep/end tokens here.
                if i == len(segments) - 1:
                    if add_end_value:
                        example_tokens.extend(self.end_value)
                        example_segment_ids.extend([i] * len(self.end_value))
                else:
                    example_tokens.extend(self.sep_value)
                    example_segment_ids.extend([i] * len(self.sep_value))

            token_ids.append(example_tokens)
            segment_ids.append(example_segment_ids)

        return token_ids, segment_ids

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

    def call(
        self,
        inputs,
        sequence_length=None,
        add_start_value=True,
        add_end_value=True,
    ):
        inputs, batched = self._canonicalize_inputs(inputs)

        segments = self._trim_inputs(inputs)
        token_ids, segment_ids = self._combine_inputs(
            segments,
            add_start_value=add_start_value,
            add_end_value=add_end_value,
        )
        # Pad to dense tensor output.
        sequence_length = sequence_length or self.sequence_length
        token_ids = self._pad(
            token_ids,
            pad_value=self.pad_value,
            padding_side=self.padding_side,
            sequence_length=sequence_length,
        )
        segment_ids = self._pad(
            segment_ids,
            pad_value=0,
            padding_side=self.padding_side,
            sequence_length=sequence_length,
        )
        # Remove the batch dim if added.
        if not batched:
            token_ids = token_ids[0]
            segment_ids = segment_ids[0]
        return (token_ids, segment_ids)

    def compute_output_shape(self, inputs_shape):
        if isinstance(inputs_shape[0], tuple):
            inputs_shape = inputs_shape[0]
        inputs_shape = list(inputs_shape)
        inputs_shape[-1] = self.sequence_length
        return tuple(inputs_shape)
