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

"""Perplexity metric implementation based on `keras.metrics.Metric`."""

import tensorflow as tf
from tensorflow import keras


class Perplexity(keras.metrics.Metric):
    """Perplexity metric.

    This class implements the perplexity metric. In short, this class calculates
    the cross entropy loss and takes its exponent.
    Note: This implementation is not suitable for fixed-size windows.

    Args:
        name: string. Name of the metric instance.
        dtype: string or tf.dtypes.Dtype. Precision of metric computation. If
               not specified, it defaults to tf.float32.
        from_logits: bool. If True, `y_pred` (input to `update_state()`) should
            be the logits as returned by the model. Otherwise, `y_pred` is a
            tensor of probabilities.
        pad_token_id: int. Token ID of the padding token. If provided, the mask
            is computed by this class (all padding tokens are masked while
            computing the cross entropy loss). Note that if this field is
            provided, the `sample_weight` field in `update_state()` is ignored.
        **kwargs: Other keyword arguments.

    Examples:

    1. Calculate perplexity by calling update_state() and result().
    1.1. `sample_weight`, and `pad_token_id` are not provided.
    >>> tf.random.set_seed(42)
    >>> perplexity = keras_nlp.metrics.Perplexity(name="perplexity")
    >>> target = tf.random.uniform(
    ...     shape=[2, 5],  maxval=10, dtype=tf.int32, seed=42)
    >>> logits = tf.random.uniform(shape=(2, 5, 10), seed=42)
    >>> perplexity.update_state(target, logits)
    >>> perplexity.result()
    <tf.Tensor: shape=(), dtype=float32, numpy=11.8781595>

    1.2. `sample_weight` specified (masking token with ID 0).
    >>> tf.random.set_seed(42)
    >>> perplexity = keras_nlp.metrics.Perplexity(name="perplexity")
    >>> target = tf.random.uniform(
    ...     shape=[2, 5],  maxval=10, dtype=tf.int32, seed=42)
    >>> logits = tf.random.uniform(shape=(2, 5, 10), seed=42)
    >>> sample_weight = tf.cast(
    ...     tf.math.logical_not(tf.equal(target, 0)), tf.float32)
    >>> perplexity.update_state(target, logits, sample_weight)
    >>> perplexity.result()
    <tf.Tensor: shape=(), dtype=float32, numpy=13.1128>

    2. Call perplexity directly.
    >>> tf.random.set_seed(42)
    >>> perplexity = keras_nlp.metrics.Perplexity(name="perplexity")
    >>> target = tf.random.uniform(
    ...     shape=[2, 5],  maxval=10, dtype=tf.int32, seed=42)
    >>> logits = tf.random.uniform(shape=(2, 5, 10), seed=42)
    >>> perplexity(target, logits)
    <tf.Tensor: shape=(), dtype=float32, numpy=11.8781595>

    3. Provide the padding token ID and let the class compute the mask on its
       own.
    >>> tf.random.set_seed(42)
    >>> perplexity = keras_nlp.metrics.Perplexity(
    ...     name="perplexity", pad_token_id=0)
    >>> target = tf.random.uniform(
    ...     shape=[2, 5],  maxval=10, dtype=tf.int32, seed=42)
    >>> logits = tf.random.uniform(shape=(2, 5, 10), seed=42)
    >>> perplexity(target, logits)
    <tf.Tensor: shape=(), dtype=float32, numpy=13.1128>
    """

    def __init__(
        self,
        name="perplexity",
        dtype=None,
        from_logits=False,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)

        if not tf.as_dtype(self.dtype).is_floating:
            raise ValueError(
                "`dtype` must be a floating point type. "
                f"Received: dtype={dtype}"
            )

        self._cross_entropy = keras.losses.SparseCategoricalCrossentropy(
            from_logits=from_logits, reduction="sum"
        )

        self.pad_token_id = pad_token_id

        self._aggregate_cross_entropy = self.add_weight(
            name="aggregate_cross_entropy",
            initializer="zeros",
            dtype=self.dtype,
        )
        self._number_of_samples = self.add_weight(
            name="number_of_samples", initializer="zeros", dtype=self.dtype
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true shape: (batch_size, seq_len)
        # y_pred shape: (batch_size, seq_len, vocab_size)
        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred, self.dtype)
        batch_size = tf.cast(tf.shape(y_true)[0], self.dtype)

        if self.pad_token_id is not None:
            sample_weight = tf.cast(
                tf.math.logical_not(tf.equal(y_true, self.pad_token_id)),
                self.dtype,
            )

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)

        # Calculate the Cross Entropy Loss.
        cross_entropy_value = tf.cast(
            self._cross_entropy(y_true, y_pred, sample_weight=sample_weight),
            self.dtype,
        )  # scalar

        # Divide the loss by the number of non-masked tokens
        if sample_weight is not None:
            cross_entropy_value = cross_entropy_value / tf.reduce_sum(
                sample_weight
            )  # scalar
        else:
            cross_entropy_value = cross_entropy_value / (
                tf.cast(tf.shape(y_true)[0], self.dtype)
                * tf.cast(tf.shape(y_true)[1], self.dtype)
            )  # scalar

        self._aggregate_cross_entropy.assign_add(
            batch_size * cross_entropy_value
        )
        self._number_of_samples.assign_add(batch_size)

    def result(self):
        if self._number_of_samples == 0:
            return 0.0
        perplexity_score = tf.exp(
            self._aggregate_cross_entropy / self._number_of_samples
        )
        return perplexity_score

    def reset_state(self):
        self._aggregate_cross_entropy.assign(0.0)
        self._number_of_samples.assign(0.0)
