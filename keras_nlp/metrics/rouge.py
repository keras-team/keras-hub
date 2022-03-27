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

"""ROUGE-N metric implementation based on `keras.metrics.Metric`."""

import tensorflow as tf
from tensorflow import keras

from keras_nlp.metrics.utils import get_intersection_of_ngrams
from keras_nlp.metrics.utils import get_ngrams


class RougeN(keras.metrics.Metric):
    """ROUGE-N metric.
    This class implements the Rouge-N metric, as described in the paper
    "ROUGE: A Package for Automatic Evaluation of Summaries"
    (https://aclanthology.org/W04-1013/).

    Args:
        name: string. Name of the metric instance.
        dtype: string or tf.dtypes.Dtype. Precision for the computation of the
            metric. If not specified, it defaults to tf.float32.
        ids_to_be_masked: [int]. Token IDs of the tokens to be masked. If
            provided, the mask is computed by this class. For example, token ID
            of the padding token can be provided here.
        order: int. Defaults to 2. The n-gram order. For example, if the order
            is 2, then bigrams are considered.
        epsilon: float. Defaults to 1e-8. Small value to avoid division by zero
            during F1 Score computation.
        **kwargs: Other keyword arguments.
    """

    def __init__(
        self,
        name="rouge_n",
        dtype=None,
        ids_to_be_masked=None,
        order=2,
        epsilon=1e-8,
        **kwargs,
    ):
        super(RougeN, self).__init__(name=name, dtype=dtype, **kwargs)
        self._dtype = dtype if dtype else tf.float32

        self.ids_to_be_masked = ids_to_be_masked

        self.order = order
        self.epsilon = epsilon

        self.rouge_n = self.add_weight(name="rouge_n", initializer="zeros")
        self.number_of_samples = self.add_weight(
            name="number_of_samples", initializer="zeros"
        )

    def _get_f1_score(self, y_true_count, y_pred_count, overlap_count):
        if y_pred_count == 0:
            precision = 0.0
        else:
            precision = tf.cast(overlap_count / y_pred_count, tf.float32)
        if y_true_count == 0:
            recall = 0.0
        else:
            recall = tf.cast(overlap_count / y_true_count, tf.float32)
        return (2 * precision * recall) / (precision + recall + self.epsilon)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true shape: (bsz, seq_len), y_pred shape: (bsz, seq_len)
        if tf.rank(y_true) != 2:
            raise ValueError(
                f"`y_true` must have rank 2. Found rank {tf.rank(y_true)}"
            )
        if tf.rank(y_pred) != 2:
            raise ValueError(
                f"`y_pred` must have rank 2. Found rank {tf.rank(y_pred)}"
            )

        if (
            tf.reduce_sum(
                tf.cast(
                    tf.equal(tf.shape(y_true), tf.shape(y_pred)),
                    tf.int32,
                )
            )
            != 2
        ):
            raise ValueError(
                f"`y_true` and `y_pred` must have the same shape. Full shape "
                f"received for `y_true`: {tf.shape(y_true)}. Full shape "
                f"received for `y_pred`: {tf.shape(y_pred)}"
            )

        # Typecast the tensors to self._dtype.
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)

        total_rouge_score = 0.0
        bsz = tf.shape(y_true)[0]
        for batch_dim in range(bsz):
            _batch_y_true = y_true[batch_dim]
            _batch_y_pred = y_pred[batch_dim]

            if self.ids_to_be_masked:
                for mask_id in self.ids_to_be_masked:
                    _mask_batch_y_true = tf.math.not_equal(
                        _batch_y_true, mask_id
                    )
                    _batch_y_true = tf.boolean_mask(
                        _batch_y_true, _mask_batch_y_true
                    )

                    _mask_batch_y_pred = tf.math.not_equal(
                        _batch_y_pred, mask_id
                    )
                    _batch_y_pred = tf.boolean_mask(
                        _batch_y_pred, _mask_batch_y_pred
                    )

            _batch_y_true_ngrams, y_true_count = get_ngrams(
                _batch_y_true, self.order
            )
            _batch_y_pred_ngrams, y_pred_count = get_ngrams(
                _batch_y_pred, self.order
            )
            _, _overlap_ct = get_intersection_of_ngrams(
                _batch_y_true_ngrams,
                y_true_count,
                _batch_y_pred_ngrams,
                y_pred_count,
            )
            total_rouge_score += self._get_f1_score(
                y_true_count, y_pred_count, _overlap_ct
            )
        self.rouge_n.assign_add(total_rouge_score)
        self.number_of_samples.assign_add(tf.cast(bsz, tf.float32))

    def result(self):
        if self.number_of_samples == 0:
            return 0.0
        return self.rouge_n / self.number_of_samples

    def reset_state(self):
        self.rouge_n.assign(0.0)
        self.number_of_samples.assign(0.0)
