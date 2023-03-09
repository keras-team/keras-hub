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
import tensorflow as tf
from tensorflow import keras

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.utils.tf_utils import assert_tf_text_installed

try:
    import tensorflow_text as tf_text
except ImportError:
    tf_text = None


# TODO: This is a temporary, unexported layer until we find a way to make the
# `MultiSegmentPacker` layer more generic.
@keras_nlp_export("keras_nlp.models.RobertaMultiSegmentPacker")
class RobertaMultiSegmentPacker(keras.layers.Layer):
    """Packs multiple sequences into a single fixed width model input.

    This layer packs multiple input sequences into a single fixed width sequence
    containing start and end delimiters, forming a dense input suitable for a
    classification task for RoBERTa.

    Takes as input a list or tuple of token segments. The layer will process
    inputs as follows:
     - Truncate all input segments to fit within `sequence_length` according to
       the `truncate` strategy.
     - Concatenate all input segments, adding a single `start_value` at the
       start of the entire sequence, `[end_value, end_value]` at the end of
       each segment save the last, and a single `end_value` at the end of the
       entire sequence.
     - Pad the resulting sequence to `sequence_length` using `pad_tokens`.

    Input should be either a `tf.RaggedTensor` or a dense `tf.Tensor`, and
    either rank-1 or rank-2.

    Please refer to the arguments of `keras_nlp.layers.MultiSegmentPacker` for
    more details.
    """

    def __init__(
        self,
        sequence_length,
        start_value,
        end_value,
        pad_value=None,
        truncate="round_robin",
        **kwargs,
    ):
        assert_tf_text_installed(self.__class__.__name__)

        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        if truncate not in ("round_robin", "waterfall"):
            raise ValueError(
                "Only 'round_robin' and 'waterfall' algorithms are "
                "supported. Received %s" % truncate
            )
        self.truncate = truncate
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
                "truncate": self.truncate,
            }
        )
        return config

    def _trim_inputs(self, inputs):
        """Trim inputs to desired length."""
        # Special tokens include the start token at the beginning of the
        # sequence, two `end_value` at the end of every segment save the last,
        # and the `end_value` at the end of the sequence.
        num_special_tokens = 2 * len(inputs)
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

    def _combine_inputs(self, segments):
        """Combine inputs with start and end values added."""
        dtype = segments[0].dtype
        batch_size = segments[0].nrows()

        start_value = tf.convert_to_tensor(self.start_value, dtype=dtype)
        end_value = tf.convert_to_tensor(self.end_value, dtype=dtype)

        start_column = tf.fill((batch_size, 1), start_value)
        end_column = tf.fill((batch_size, 1), end_value)

        segments_to_combine = []
        for i, seg in enumerate(segments):
            segments_to_combine.append(start_column if i == 0 else end_column)
            segments_to_combine.append(seg)
            segments_to_combine.append(end_column)

        token_ids = tf.concat(segments_to_combine, 1)
        return token_ids

    def call(self, inputs):
        def to_ragged(x):
            return tf.RaggedTensor.from_tensor(x[tf.newaxis, :])

        # If rank 1, add a batch dim.
        rank_1 = inputs[0].shape.rank == 1
        if rank_1:
            inputs = [to_ragged(x) for x in inputs]

        segments = self._trim_inputs(inputs)
        token_ids = self._combine_inputs(segments)
        # Pad to dense tensor output.
        shape = tf.cast([-1, self.sequence_length], tf.int64)
        token_ids = token_ids.to_tensor(
            shape=shape, default_value=self.pad_value
        )
        # Remove the batch dim if added.
        if rank_1:
            token_ids = tf.squeeze(token_ids, 0)

        return token_ids
