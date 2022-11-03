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

"""A base class for models including preprocessing."""

import math

import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras_nlp.utils.keras_utils import pack_x_y_sample_weight

try:
    import pandas as pd
except ImportError:
    pd = None


def convert_inputs_to_dataset(
    x=None,
    y=None,
    sample_weight=None,
    batch_size=None,
):
    """Convert inputs to a `tf.data.Dataset`."""
    if isinstance(x, tf.data.Dataset):
        if y is not None:
            raise ValueError(
                f"When `x` is a `tf.data.Dataset`, please do not provide "
                f"`y`. Received: `type(y)={type(y)}`."
            )
        if sample_weight is not None:
            raise ValueError(
                f"When `x` is a `tf.data.Dataset`, please do not provide "
                f"`sample_weight`. Received: "
                f"`type(sample_weight)={type(sample_weight)}`."
            )
        if batch_size is not None:
            raise ValueError(
                f"When `x` is a `tf.data.Dataset`, please do not provide "
                f"`batch_size`. Received: "
                f"`type(batch_size)={type(batch_size)}`."
            )
        return x

    inputs = keras.utils.pack_x_y_sample_weight(x, y, sample_weight)
    return tf.data.Dataset.from_tensor_slices(inputs).batch(batch_size or 32)


def train_validation_split(data, validation_split):
    """Mimic the `validation_split` of `keras.Model.fit()`."""
    flat_data = list(filter(lambda t: t is not None, tf.nest.flatten(data)))

    supported_types = (tf.Tensor, np.ndarray)
    if pd is not None:
        supported_types += (pd.Series, pd.DataFrame)

    for t in flat_data:
        if not isinstance(t, supported_types):
            raise ValueError(
                f"`validation_split` is only supported for Tensors or NumPy "
                f"arrays, found following types in the input: {type(t)}"
            )

    if not flat_data:
        return data, data

    # Assumes all data have the same batch shape or are `None`.
    batch_dim = int(flat_data[0].shape[0])
    split_at = int(math.floor(batch_dim * (1.0 - validation_split)))

    if split_at == 0 or split_at == batch_dim:
        raise ValueError(
            f"Training data contains {batch_dim} samples, which is not "
            f"sufficient to split it into a validation and training set as "
            f"specified by `validation_split={validation_split}`. Either "
            f"provide more data, or a different value for the "
            f"`validation_split` argument."
        )

    training_data = tf.nest.map_structure(
        lambda t: None if t is None else t[:split_at], data
    )
    validation_data = tf.nest.map_structure(
        lambda t: None if t is None else t[split_at:], data
    )
    return training_data, validation_data


class PipelineModel(keras.Model):
    """A model which allows automatically applying preprocessing."""

    def __init__(self, *args, include_preprocessing=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_preprocessing = include_preprocessing

    def preprocess_features(self, x):
        """An overridable function which preprocesses features."""
        return x

    def preprocess_labels(self, y):
        """An overridable function which preprocesses labels."""
        return y

    def preprocess_samples(self, x, y=None, sample_weight=None):
        """An overridable function which preprocesses entire samples."""
        x = self.preprocess_features(x)
        if y is not None:
            y = self.preprocess_labels(y)
        return pack_x_y_sample_weight(x, y, sample_weight)

    # ========================================================================
    # Below are overrides to keras.Model methods to apply the functions above.
    # ========================================================================
    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        sample_weight=None,
        validation_data=None,
        validation_split=None,
        **kwargs,
    ):
        if validation_split and validation_data is None:
            (x, y, sample_weight), validation_data = train_validation_split(
                (x, y, sample_weight), validation_split=validation_split
            )

        x = convert_inputs_to_dataset(x, y, sample_weight, batch_size)
        if self.include_preprocessing:
            x = x.map(self.preprocess_samples, tf.data.AUTOTUNE)
            x = x.prefetch(tf.data.AUTOTUNE)

        if validation_data is not None:
            if not isinstance(validation_data, tf.data.Dataset):
                (vx, vy, vsw) = keras.utils.unpack_x_y_sample_weight(
                    validation_data
                )
                validation_data = convert_inputs_to_dataset(
                    vx, vy, vsw, batch_size
                )

        return super().fit(
            x=x,
            y=None,
            batch_size=None,
            sample_weight=None,
            validation_data=validation_data,
            **kwargs,
        )

    def evaluate(
        self,
        x=None,
        y=None,
        batch_size=None,
        sample_weight=None,
        **kwargs,
    ):
        # TODO: we can't support a cached eval dataset until we make changes to
        # the upstream model. Otherwise we would cache the raw dataset.
        kwargs.pop("_use_cached_eval_dataset", None)
        x = convert_inputs_to_dataset(x, y, sample_weight, batch_size)
        if self.include_preprocessing:
            x = x.map(self.preprocess_samples, tf.data.AUTOTUNE)
            x = x.prefetch(tf.data.AUTOTUNE)
        return super().evaluate(
            x=x,
            y=None,
            batch_size=None,
            **kwargs,
        )

    def predict(
        self,
        x=None,
        batch_size=None,
        **kwargs,
    ):
        x = convert_inputs_to_dataset(x, None, None, batch_size)
        if self.include_preprocessing:
            x = x.map(self.preprocess_samples, tf.data.AUTOTUNE)
            x = x.prefetch(tf.data.AUTOTUNE)

        return super().predict(
            x=x,
            batch_size=None,
            **kwargs,
        )

    def __call__(self, inputs, include_preprocessing=None, **kwargs):
        # We don't trace if `including_preprocessing` is `False`, or we are
        # currently tracing a functional model.
        flat_inputs = tf.nest.flatten(inputs)
        tracing = any([type(t).__name__ == "KerasTensor" for t in flat_inputs])
        if include_preprocessing is None:
            include_preprocessing = self.include_preprocessing
        if include_preprocessing and not tracing:
            data = self.preprocess_samples(inputs)
            inputs, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        return super().__call__(inputs, **kwargs)

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True, include_preprocessing=False)
            loss = self.compute_loss(x, y, y_pred, sample_weight)
        self._validate_target_and_loss(y, loss)
        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(x, y, y_pred, sample_weight)

    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)

        y_pred = self(x, training=False, include_preprocessing=False)
        # Updates stateful loss metrics.
        self.compute_loss(x, y, y_pred, sample_weight)
        return self.compute_metrics(x, y, y_pred, sample_weight)

    def predict_step(self, data):
        x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        return self(x, training=False, include_preprocessing=False)

    def train_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        **kwargs,
    ):
        if self.include_preprocessing:
            data = self.preprocess_samples(x, y, sample_weight)
            x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        return super().train_on_batch(
            x=x,
            y=y,
            sample_weight=sample_weight,
            **kwargs,
        )

    def test_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        **kwargs,
    ):
        if self.include_preprocessing:
            data = self.preprocess_samples(x, y, sample_weight)
            x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        return super().test_on_batch(
            x=x,
            y=y,
            sample_weight=sample_weight,
            **kwargs,
        )

    def predict_on_batch(
        self,
        x,
        **kwargs,
    ):
        if self.include_preprocessing:
            data = self.preprocess_samples(x)
            x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        return super().predict_on_batch(
            x=x,
            **kwargs,
        )
