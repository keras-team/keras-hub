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

import functools
import math

import keras
from keras import ops
from keras import tree

from keras_hub.src.utils.tensor_utils import is_tensor_type

try:
    import tensorflow as tf
except ImportError:
    tf = None


def _convert_inputs_to_dataset(
    x=None,
    y=None,
    sample_weight=None,
    batch_size=None,
):
    """Convert inputs to a `tf.data.Dataset`.

    This is a stand in for the `TensorLikeDataAdapter` in core Keras.
    """
    if isinstance(x, tf.data.Dataset):
        if y is not None:
            raise ValueError(
                "When `x` is a `tf.data.Dataset`, please do not provide "
                f"`y`. Received: `type(y)={type(y)}`."
            )
        if sample_weight is not None:
            raise ValueError(
                "When `x` is a `tf.data.Dataset`, please do not provide "
                "`sample_weight`. Received: "
                f"`type(sample_weight)={type(sample_weight)}`."
            )
        if batch_size is not None:
            raise ValueError(
                "When `x` is a `tf.data.Dataset`, please do not provide "
                "`batch_size`. Received: "
                f"`type(batch_size)={type(batch_size)}`."
            )
        return x

    inputs = keras.utils.pack_x_y_sample_weight(x, y, sample_weight)
    try:

        def convert(x):
            if isinstance(x, (tf.Tensor, tf.RaggedTensor)):
                return x
            if hasattr(x, "__array__"):
                return ops.convert_to_numpy(x)
            return x

        inputs = tree.map_structure(convert, inputs)
        ds = tf.data.Dataset.from_tensor_slices(inputs)
    except ValueError as e:
        # If our inputs are unbatched, re-raise with a more friendly error
        # message the default from tf.data. We expect this to come up with
        # some frequency, so it's important to have a good sign post here.
        if "only supported for rank >= 1" in str(e):
            raise ValueError(
                "`x`, `y`, and `sample_weight` must have a batch dimension "
                "when calling `fit()`, `evaluate()`, and `predict()`. Received "
                "an input with rank 0. Please add an outer dimension to your "
                "input, e.g., wrap it in a list."
            ) from e
        raise e

    return ds.batch(batch_size or 32)


def _train_validation_split(arrays, validation_split):
    """Split arrays into train and validation subsets in deterministic order.

    This is copied directly from core Keras.
    """

    def _can_split(t):
        return is_tensor_type(t) or t is None

    flat_arrays = tree.flatten(arrays)
    unsplitable = [type(t) for t in flat_arrays if not _can_split(t)]
    if unsplitable:
        raise ValueError(
            "`validation_split` is only supported for Tensors or NumPy "
            "arrays, found following types in the input: {}".format(unsplitable)
        )

    if all(t is None for t in flat_arrays):
        return arrays, arrays

    first_non_none = None
    for t in flat_arrays:
        if t is not None:
            first_non_none = t
            break

    # Assumes all arrays have the same batch shape or are `None`.
    batch_dim = int(first_non_none.shape[0])
    split_at = int(math.floor(batch_dim * (1.0 - validation_split)))

    if split_at == 0 or split_at == batch_dim:
        raise ValueError(
            "Training data contains {batch_dim} samples, which is not "
            "sufficient to split it into a validation and training set as "
            "specified by `validation_split={validation_split}`. Either "
            "provide more data, or a different value for the "
            "`validation_split` argument.".format(
                batch_dim=batch_dim, validation_split=validation_split
            )
        )

    def _split(t, start, end):
        if t is None:
            return t
        return t[start:end]

    train_arrays = tree.map_structure(
        functools.partial(_split, start=0, end=split_at), arrays
    )
    val_arrays = tree.map_structure(
        functools.partial(_split, start=split_at, end=batch_dim), arrays
    )

    return train_arrays, val_arrays


@keras.saving.register_keras_serializable(package="keras_hub")
class PipelineModel(keras.Model):
    """A model which allows automatically applying preprocessing."""

    def __init__(self, *args, **kwargs):
        # Workaround for https://github.com/keras-team/keras/issues/17270
        # Reset any attempt to overwrite this classes base class to this class
        # can continue to be used for functional and non-functional models.
        PipelineModel.__bases__ = (keras.Model,)
        super().__init__(*args, **kwargs)

    def preprocess_samples(self, x, y=None, sample_weight=None):
        """An overridable function which preprocesses entire samples."""
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

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
            (x, y, sample_weight), validation_data = _train_validation_split(
                (x, y, sample_weight), validation_split=validation_split
            )

        x = _convert_inputs_to_dataset(x, y, sample_weight, batch_size)
        x = x.map(
            self.preprocess_samples, num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)

        if validation_data is not None:
            if not isinstance(validation_data, tf.data.Dataset):
                (vx, vy, vsw) = keras.utils.unpack_x_y_sample_weight(
                    validation_data
                )
                validation_data = _convert_inputs_to_dataset(
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
        # During `fit()`, `keras.Model` attempts to cache the validation
        # dataset and ignores the values for `x`, `y`, and `sample_weight`.
        # We don't want that behavior here, as the validation dataset still
        # needs preprocessing.
        kwargs.pop("_use_cached_eval_dataset", None)
        x = _convert_inputs_to_dataset(x, y, sample_weight, batch_size)
        x = x.map(
            self.preprocess_samples, num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
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
        x = _convert_inputs_to_dataset(x, None, None, batch_size)
        x = x.map(
            self.preprocess_samples, num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        return super().predict(
            x=x,
            batch_size=None,
            **kwargs,
        )

    def train_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        **kwargs,
    ):
        data = self.preprocess_samples(x, y, sample_weight)
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        x = ops.convert_to_tensor(x)
        if y is not None:
            y = ops.convert_to_tensor(y)
        if sample_weight is not None:
            sample_weight = ops.convert_to_tensor(sample_weight)
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
        data = self.preprocess_samples(x, y, sample_weight)
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        x = ops.convert_to_tensor(x)
        if y is not None:
            y = ops.convert_to_tensor(y)
        if sample_weight is not None:
            sample_weight = ops.convert_to_tensor(sample_weight)
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
        data = self.preprocess_samples(x)
        x, _, _ = keras.utils.unpack_x_y_sample_weight(data)
        x = ops.convert_to_tensor(x)
        return super().predict_on_batch(
            x=x,
            **kwargs,
        )
