# Copyright 2024 The KerasHub Authors
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

import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.utils.tensor_utils import is_float_dtype


@keras_hub_export("keras_hub.metrics.Perplexity")
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
               not specified, it defaults to `"float32"`.
        name: string. Name of the metric instance.
        **kwargs: Other keyword arguments.

    Examples:

    1. Calculate perplexity by calling update_state() and result().
    1.1. `sample_weight`, and `mask_token_id` are not provided.
    >>> np.random.seed(42)
    >>> perplexity = keras_hub.metrics.Perplexity(name="perplexity")
    >>> target = np.random.randint(10, size=[2, 5])
    >>> logits = np.random.uniform(size=(2, 5, 10))
    >>> perplexity.update_state(target, logits)
    >>> perplexity.result()
    <tf.Tensor: shape=(), dtype=float32, numpy=14.352535>

    1.2. `sample_weight` specified (masking token with ID 0).
    >>> np.random.seed(42)
    >>> perplexity = keras_hub.metrics.Perplexity(name="perplexity")
    >>> target = np.random.randint(10, size=[2, 5])
    >>> logits = np.random.uniform(size=(2, 5, 10))
    >>> sample_weight = (target != 0).astype("float32")
    >>> perplexity.update_state(target, logits, sample_weight)
    >>> perplexity.result()
    <tf.Tensor: shape=(), dtype=float32, numpy=14.352535>

    2. Call perplexity directly.
    >>> np.random.seed(42)
    >>> perplexity = keras_hub.metrics.Perplexity(name="perplexity")
    >>> target = np.random.randint(10, size=[2, 5])
    >>> logits = np.random.uniform(size=(2, 5, 10))
    >>> perplexity(target, logits)
    <tf.Tensor: shape=(), dtype=float32, numpy=14.352535>

    3. Provide the padding token ID and let the class compute the mask on its
       own.
    >>> np.random.seed(42)
    >>> perplexity = keras_hub.metrics.Perplexity(mask_token_id=0)
    >>> target = np.random.randint(10, size=[2, 5])
    >>> logits = np.random.uniform(size=(2, 5, 10))
    >>> perplexity(target, logits)
    <tf.Tensor: shape=(), dtype=float32, numpy=14.352535>
    """

    def __init__(
        self,
        from_logits=False,
        mask_token_id=None,
        dtype="float32",
        name="perplexity",
        **kwargs,
    ):
        if not is_float_dtype(dtype):
            raise ValueError(
                "`dtype` must be a floating point type. "
                f"Received: dtype={dtype}"
            )

        super().__init__(name=name, dtype=dtype, **kwargs)

        self._crossentropy = keras.losses.SparseCategoricalCrossentropy(
            from_logits=from_logits, reduction="sum"
        )

        self.from_logits = from_logits
        self.mask_token_id = mask_token_id

        self._aggregate_crossentropy = self.add_weight(
            shape=(),
            initializer="zeros",
            dtype=self.dtype,
            name="aggregate_crossentropy",
        )
        self._number_of_samples = self.add_weight(
            shape=(),
            initializer="zeros",
            dtype=self.dtype,
            name="number_of_samples",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true shape: (batch_size, seq_len)
        # y_pred shape: (batch_size, seq_len, vocab_size)
        y_true = ops.cast(y_true, self.dtype)
        y_pred = ops.cast(y_pred, self.dtype)

        if sample_weight is not None:
            sample_weight = ops.cast(sample_weight, self.dtype)

        batch_size = ops.cast(ops.shape(y_true)[0], self.dtype)

        if self.mask_token_id is not None:
            mask = ops.cast(
                ops.logical_not(ops.equal(y_true, self.mask_token_id)),
                self.dtype,
            )
            if sample_weight is None:
                sample_weight = mask
            else:
                sample_weight = ops.multiply(mask, sample_weight)

        # Calculate the Cross Entropy Loss.
        crossentropy_value = ops.cast(
            self._crossentropy(y_true, y_pred, sample_weight=sample_weight),
            self.dtype,
        )  # scalar

        # Divide the loss by the number of non-masked tokens
        if sample_weight is not None:
            crossentropy_value = crossentropy_value / ops.sum(
                sample_weight
            )  # scalar
        else:
            crossentropy_value = crossentropy_value / (
                ops.cast(ops.shape(y_true)[0], self.dtype)
                * ops.cast(ops.shape(y_true)[1], self.dtype)
            )  # scalar

        self._aggregate_crossentropy.assign_add(batch_size * crossentropy_value)
        self._number_of_samples.assign_add(batch_size)

    def result(self):
        perplexity_score = ops.where(
            ops.equal(ops.convert_to_tensor(self._number_of_samples), 0),
            0,
            ops.exp(self._aggregate_crossentropy / self._number_of_samples),
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
