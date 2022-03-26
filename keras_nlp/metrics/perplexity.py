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
        dtype: string or tf.dtypes.Dtype. Data type of y_pred, to which all
            tensors are also cast, for computation. If not specified, it
            defaults to tf.float32.
        from_logits: bool. If True, `y_pred` (input to `update_state()`) should
            be the logits as returned by the model. Otherwise, `y_pred` is a
            tensor of probabilities.
        pad_token_id: int. Token ID of the padding token. If provided, the mask
            is computed by this class (all padding tokens are masked while
            computing the cross entropy loss). Note that if this field is
            provided, the `sample_weight` field in `update_state()` is ignored.
        **kwargs: Other keyword arguments.

    Examples:

    ```python
    # 1. update_state() and result()
    perplexity = keras_nlp.metrics.Perplexity(name="perplexity")
    target = tf.experimental.numpy.random.randint(low=0, high=10, size=(2, 5))
    logits = tf.random.uniform(shape=(2, 5, 10))

    # 1.1 sample_weight not specified.
    perplexity.update_state(target, logits)
    print(perplexity.result())

    # 1.2 sample_weight specified.
    # mask token 0
    sample_weight = tf.cast(tf.math.logical_not(tf.equal(y_true, 0)),
                            tf.float32)
    perplexity.update_state(target, logits, sample_weight)
    print(perplexity.result())

    # 2. Call perplexity directly.
    perplexity = keras_nlp.metrics.Perplexity(name="perplexity")
    print(perplexity(target, logits))

    # 3. Provide the padding token ID and let the class compute the mask on its
    # own.
    perplexity = keras_nlp.metrics.Perplexity(name="perplexity", pad_token_id=0)
    print(perplexity(target, logits))
    ```
    """

    def __init__(
        self,
        name="perplexity",
        dtype=None,
        from_logits=False,
        pad_token_id=None,
        **kwargs
    ):
        super(Perplexity, self).__init__(name=name, dtype=dtype, **kwargs)
        self._dtype = dtype if dtype else tf.float32

        self.cross_entropy_loss = keras.losses.SparseCategoricalCrossentropy(
            from_logits=from_logits, reduction="sum"
        )

        self.pad_token_id = pad_token_id

        self.aggregate_cross_entropy_loss = self.add_weight(
            name="aggregate_cross_entropy_loss",
            initializer="zeros",
            dtype=self._dtype,
        )
        self.number_of_samples = self.add_weight(
            name="number_of_samples", initializer="zeros", dtype=self._dtype
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true shape: (bsz, seq_len), y_pred shape: (bsz, seq_len, vocab_size)
        y_pred = tf.cast(y_pred, self._dtype)
        bsz = tf.cast(tf.shape(y_true)[0], self._dtype)

        if self.pad_token_id:
            sample_weight = tf.cast(
                tf.math.logical_not(tf.equal(y_true, 0)), self._dtype
            )

        # Reshape y_true and y_pred.
        y_true = tf.reshape(y_true, [-1])  # (bsz * seq_len,)
        y_pred = tf.reshape(
            y_pred, [-1, y_pred.shape[-1]]
        )  # (bsz * seq_len, vocab_size)

        # Calculate the Cross Entropy Loss.
        loss_value = tf.cast(
            self.cross_entropy_loss(
                y_true, y_pred, sample_weight=sample_weight
            ),
            self._dtype,
        )  # scalar

        # Divide the loss by the number of non-masked tokens
        if sample_weight:
            loss_value = loss_value / tf.reduce_sum(
                tf.reshape(sample_weight, [-1])
            )  # scalar
        else:
            loss_value = loss_value / tf.cast(
                tf.shape(y_true)[0], self._dtype
            )  # scalar

        self.aggregate_cross_entropy_loss.assign_add(bsz * loss_value)
        self.number_of_samples.assign_add(bsz)

    def result(self):
        if self.number_of_samples == 0:
            return 0.0
        perplexity_score = tf.exp(
            self.aggregate_cross_entropy_loss / self.number_of_samples
        )
        return perplexity_score

    def reset_state(self):
        self.aggregate_cross_entropy_loss.assign(0.0)
        self.number_of_samples.assign(0.0)
