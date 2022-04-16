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

"""ROUGE-L metric implementation based on `keras.metrics.Metric`."""

import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow import keras


class RougeL(keras.metrics.Metric):
    """ROUGE-L metric.

    This class implements the ROUGE-L metric.

    Args:
        alpha: float. `alpha` is used as the weight for the
            harmonic mean of precision and recall. A value of 0 means recall is
            more important and a value of 1 means precision is more important
            (same behaviour as
            https://www.tensorflow.org/text/api_docs/python/text/metrics/rouge_l).
        metric_type: string. One of "precision", "recall", "f1_score". Defaults
            to "f1_score".
        mask_token_id: int. ID of the token to be masked. If provided, the mask
            is computed for this class. Note that if this field is provided, and
            if the `sample_weight` field in `update_state()` is also provided,
            we will compute the final `sample_weight` as the element-wise
            product of the mask and the `sample_weight`. In the product, any
            value >= 1 will be treated as True, and False, otherwise, for
            masking.
        dtype: string or tf.dtypes.Dtype. Precision of metric computation. If
               not specified, it defaults to tf.float32.
        name: string. Name of the metric instance.
        **kwargs: Other keyword arguments.

    Examples:

    """

    def __init__(
        self,
        alpha=0.5,
        metric_type="f1_score",
        mask_token_id=None,
        dtype=None,
        name="rouge_l",
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)

        if not tf.as_dtype(self.dtype).is_floating:
            raise ValueError(
                "`dtype` must be a floating point type. "
                f"Received: dtype={dtype}"
            )

        if metric_type not in ["precision", "recall", "f1_score"]:
            raise ValueError(
                "`metric_type` must be one of 'precision', 'recall', "
                "'f1_score'. Received: metric_type={metric_type}"
            )

        self.alpha = alpha
        self.metric_type = metric_type
        self.mask_token_id = mask_token_id

        self._rouge_l_score = self.add_weight(
            name="rouge_l_score",
            initializer="zeros",
            dtype=self.dtype,
        )
        self._number_of_samples = self.add_weight(
            name="number_of_samples", initializer="zeros", dtype=self.dtype
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Both y_true and y_pred have shape: [batch_size, seq_len]. Note that
        # they can also be ragged tensors with shape [num_samples, (seq_len)].

        # If the input tensors are not ragged tensors, convert them to ragged
        # tensors. `tf_text.metrics.rouge_l` expects ragged tensors.
        if not isinstance(y_true, tf.RaggedTensor):
            y_true = tf.RaggedTensor.from_tensor(y_true)
        if not isinstance(y_pred, tf.RaggedTensor):
            y_pred = tf.RaggedTensor.from_tensor(y_pred)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)

        batch_size = tf.cast(y_true.nrows(), self.dtype)

        if self.mask_token_id is not None:
            mask = tf.cast(
                tf.math.logical_not(tf.equal(y_true, self.mask_token_id)),
                self.dtype,
            )
            if sample_weight is None:
                sample_weight = mask
            else:
                sample_weight = tf.multiply(mask, sample_weight)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.bool)

            # Apply mask to both tensors.
            y_true = tf.ragged.boolean_mask(y_true, sample_weight)
            y_pred = tf.ragged.boolean_mask(y_pred, sample_weight)

        f1_scores, precisions, recalls = rouge_l(
            y_true, y_pred, alpha=self.alpha
        )
        if self.metric_type == "precision":
            scores = precisions
        elif self.metric_type == "recall":
            scores = recalls
        else:
            scores = f1_scores
        self._rouge_l_score.assign_add(tf.reduce_sum(scores))
        self._number_of_samples.assign_add(batch_size)

    def result(self):
        if self._number_of_samples == 0:
            return 0.0
        rouge_l_score = self._rouge_l_score / self._number_of_samples
        return rouge_l_score

    def reset_state(self):
        self._rouge_l_score.assign(0.0)
        self._number_of_samples.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "alpha": self.alpha,
                "metric_type": self.metric_type,
                "mask_token_id": self.mask_token_id,
            }
        )
        return config


def rouge_l(y_true, y_pred, alpha=0.5):
    """
    Computes the ROUGE-L score.
    Args:
        y_true: tf.RaggedTensor. The reference summaries.
        y_pred: tf.RaggedTensor. The generated summaries.
        alpha: float. Defaults to 0.5. `alpha` is used as the weight for the
            harmonic mean of precision and recall. A value of 0 means recall is
            more important and a value of 1 means precision is more important
            (same behaviour as
            https://www.tensorflow.org/text/api_docs/python/text/metrics/rouge_l).

    Returns:
        (f1_scores, precisions, recalls): Tuple of tf.Tensor. The f1_scores,
            precisions and recalls are returned for every sample.
    """
    f1_scores, precisions, recalls = tf_text.metrics.rouge_l(
        y_true, y_pred, alpha=alpha
    )
    return f1_scores, precisions, recalls
