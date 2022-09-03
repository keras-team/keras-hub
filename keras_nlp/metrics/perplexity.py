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

"""Perplexity metric."""

import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="keras_nlp")
class Perplexity(keras.metrics.Metric):
    """Perplexity metric.

    This class implements the perplexity metric. In short, this class calculates
    the cross entropy loss and takes its exponent.
    Note: This implementation is not suitable for fixed-size windows.

    Args:
        from_logits: bool. If True, `y_pred` (input to `update_state()`) should
            be the logits as returned by the model. Otherwise, `y_pred` is a
            tensor of probabilities.
        mask_token_id: int. ID of the token to be masked. If provided, the mask
            is computed for this class. Note that if this field is provided, and
            if the `sample_weight` field in `update_state()` is also provided,
            we will compute the final `sample_weight` as the element-wise
            product of the mask and the `sample_weight`.
        dtype: string or tf.dtypes.Dtype. Precision of metric computation. If
               not specified, it defaults to tf.float32.
        name: string. Name of the metric instance.
        **kwargs: Other keyword arguments.

    Examples:

    1. Calculate perplexity by calling update_state() and result().
    1.1. `sample_weight`, and `mask_token_id` are not provided.
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
    ...     name="perplexity", mask_token_id=0)
    >>> target = tf.random.uniform(
    ...     shape=[2, 5],  maxval=10, dtype=tf.int32, seed=42)
    >>> logits = tf.random.uniform(shape=(2, 5, 10), seed=42)
    >>> perplexity(target, logits)
    <tf.Tensor: shape=(), dtype=float32, numpy=13.1128>
    """

    def __init__(
        self,
        from_logits=False,
        mask_token_id=None,
        dtype=None,
        name="perplexity",
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)

        if not tf.as_dtype(self.dtype).is_floating:
            raise ValueError(
                "`dtype` must be a floating point type. "
                f"Received: dtype={dtype}"
            )

        self._crossentropy = keras.losses.SparseCategoricalCrossentropy(
            from_logits=from_logits, reduction="sum"
        )

        self.from_logits = from_logits
        self.mask_token_id = mask_token_id

        self._aggregate_crossentropy = self.add_weight(
            name="aggregate_crossentropy",
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

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)

        batch_size = tf.cast(tf.shape(y_true)[0], self.dtype)

        if self.mask_token_id is not None:
            mask = tf.cast(
                tf.math.logical_not(tf.equal(y_true, self.mask_token_id)),
                self.dtype,
            )
            if sample_weight is None:
                sample_weight = mask
            else:
                sample_weight = tf.multiply(mask, sample_weight)

        # Calculate the Cross Entropy Loss.
        crossentropy_value = tf.cast(
            self._crossentropy(y_true, y_pred, sample_weight=sample_weight),
            self.dtype,
        )  # scalar

        # Divide the loss by the number of non-masked tokens
        if sample_weight is not None:
            crossentropy_value = crossentropy_value / tf.reduce_sum(
                sample_weight
            )  # scalar
        else:
            crossentropy_value = crossentropy_value / (
                tf.cast(tf.shape(y_true)[0], self.dtype)
                * tf.cast(tf.shape(y_true)[1], self.dtype)
            )  # scalar

        self._aggregate_crossentropy.assign_add(batch_size * crossentropy_value)
        self._number_of_samples.assign_add(batch_size)

    def result(self):
        if self._number_of_samples == 0:
            return 0.0
        perplexity_score = tf.exp(
            self._aggregate_crossentropy / self._number_of_samples
        )
        return perplexity_score

    def reset_state(self):
        self._aggregate_crossentropy.assign(0.0)
        self._number_of_samples.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "from_logits": self.from_logits,
                "mask_token_id": self.mask_token_id,
            }
        )
        return config
