import contextlib
import functools
import math
import warnings

import keras
import numpy as np
from keras import ops
from keras import tree

from keras_hub.src.utils.tensor_utils import convert_preprocessing_outputs_grain
from keras_hub.src.utils.tensor_utils import convert_to_numpy
from keras_hub.src.utils.tensor_utils import is_tensor_type

try:
    import grain
except ImportError:
    grain = None

try:
    import tensorflow as tf
except ImportError:
    tf = None


def _contains_string_data(inputs):
    """Check if any leaf in a nested structure contains string data."""
    for x in tree.flatten(inputs):
        if isinstance(x, (str, bytes)):
            return True
        if isinstance(x, np.ndarray):
            if x.dtype.kind in ("U", "S"):
                return True
            if x.dtype.kind == "O" and x.size:
                if any(isinstance(item, (str, bytes)) for item in x.flat):
                    return True
            continue
        dtype = getattr(x, "dtype", None)
        if dtype is not None and "string" in str(dtype).lower():
            return True
    return False


UNBATCHED_INPUT_ERROR = (
    "`x`, `y`, and `sample_weight` must have a batch dimension when calling "
    "`fit()`, `evaluate()`, and `predict()`. Received an input with rank 0. "
    "Please add an outer dimension to your input, e.g., wrap it in a list."
)

GRAIN_DATA_LOADER_ERROR = (
    "`PipelineModel` maps preprocessing over the dataset it is given, which "
    "a `grain.DataLoader` does not support. Pass a `grain.MapDataset` or a "
    "`grain.IterDataset` instead, e.g. "
    "`grain.MapDataset.source(source).batch(batch_size)`."
)


def _is_tf_dataset(x):
    return tf is not None and isinstance(x, tf.data.Dataset)


def _is_grain_dataset(x):
    # `grain.DataLoader` is deliberately not included. It has no `map`, so
    # preprocessing cannot be applied to one. See `_is_grain_data_loader`.
    if grain is None:
        return False
    return isinstance(x, (grain.MapDataset, grain.IterDataset))


def _is_grain_data_loader(x):
    return grain is not None and isinstance(x, grain.DataLoader)


def _map_leaves(inputs, fn):
    """Map `fn` over the leaves of `inputs`.

    Follows `tf.data.Dataset.from_tensor_slices`, where tuples and dicts are
    structure but a list is a single tensor. `keras.tree` descends into lists,
    which would split a list of strings into one leaf per string.
    """
    if isinstance(inputs, tuple):
        return tuple(_map_leaves(i, fn) for i in inputs)
    if isinstance(inputs, dict):
        return {k: _map_leaves(v, fn) for k, v in inputs.items()}
    return fn(inputs)


def _flatten_leaves(inputs):
    """List the leaves of `inputs`, with the same structure rules as above."""
    if isinstance(inputs, tuple):
        return [leaf for i in inputs for leaf in _flatten_leaves(i)]
    if isinstance(inputs, dict):
        return [leaf for v in inputs.values() for leaf in _flatten_leaves(v)]
    return [inputs]


def _is_ragged(inputs):
    """Whether any leaf is a `tf.RaggedTensor`, which needs `tf.data`."""
    if tf is None:
        return False
    return any(isinstance(t, tf.RaggedTensor) for t in _flatten_leaves(inputs))


class _TensorLikeSource:
    """A Grain source slicing a nested structure along its batch dimension.

    This is the Grain stand in for `tf.data.Dataset.from_tensor_slices`.
    """

    def __init__(self, inputs):
        leaves = _flatten_leaves(inputs)
        if not leaves or any(len(leaf.shape) == 0 for leaf in leaves):
            raise ValueError(UNBATCHED_INPUT_ERROR)
        lengths = set(int(leaf.shape[0]) for leaf in leaves)
        if len(lengths) > 1:
            # Left to Grain this would surface as an `IndexError` mid epoch.
            raise ValueError(
                "`x`, `y`, and `sample_weight` must all have the same "
                "batch dimension. Received inputs with batch sizes "
                f"{sorted(lengths)}."
            )
        self.inputs = inputs
        self.length = lengths.pop()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return _map_leaves(self.inputs, lambda leaf: leaf[index])


def _convert_to_numpy(inputs):
    """Convert every leaf of `inputs` to a numpy array.

    `tensor_utils.convert_to_numpy` handles the tf tensors that preprocessing
    layers hold regardless of the active backend, which
    `keras.ops.convert_to_numpy` cannot.
    """
    return _map_leaves(inputs, convert_to_numpy)


def _convert_strings_to_python(inputs):
    """Convert numpy string arrays to nested lists of python strings.

    Grain stacks strings into numpy arrays, but preprocessing layers take
    python `str`. See `keras_hub.utils.convert_preprocessing_inputs`.
    """

    def decode(value):
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, list):
            return [decode(v) for v in value]
        return value

    def convert(leaf):
        if isinstance(leaf, np.ndarray) and leaf.dtype.kind in ("U", "S", "O"):
            return decode(leaf.tolist())
        return leaf

    return _map_leaves(inputs, convert)


def _convert_inputs_to_dataset(
    x=None,
    y=None,
    sample_weight=None,
    batch_size=None,
):
    """Convert inputs to a batched dataset.

    This is a stand in for the `TensorLikeDataAdapter` in core Keras. Inputs
    are batched into a `grain.MapDataset`, which needs no TensorFlow runtime
    and feeds `fit()` on all Keras backends. A dataset passed in directly by
    the caller is validated and returned as is.
    """
    if _is_grain_data_loader(x):
        raise ValueError(GRAIN_DATA_LOADER_ERROR)

    if _is_tf_dataset(x) or _is_grain_dataset(x):
        kind = "tf.data.Dataset" if _is_tf_dataset(x) else "grain dataset"
        if y is not None:
            raise ValueError(
                f"When `x` is a {kind}, please do not provide "
                f"`y`. Received: `type(y)={type(y)}`."
            )
        if sample_weight is not None:
            raise ValueError(
                f"When `x` is a {kind}, please do not provide "
                "`sample_weight`. Received: "
                f"`type(sample_weight)={type(sample_weight)}`."
            )
        if batch_size is not None:
            raise ValueError(
                f"When `x` is a {kind}, please do not provide "
                "`batch_size`. Received: "
                f"`type(batch_size)={type(batch_size)}`."
            )
        return x

    inputs = keras.utils.pack_x_y_sample_weight(x, y, sample_weight)
    # Grain cannot stack ragged inputs.
    if grain is None or _is_ragged(inputs):
        return _convert_inputs_to_tf_dataset(inputs, batch_size)

    inputs = _convert_to_numpy(inputs)
    source = _TensorLikeSource(inputs)
    return grain.MapDataset.source(source).batch(batch_size or 32)


def _convert_inputs_to_tf_dataset(inputs, batch_size=None):
    """Slice and batch `inputs` with `tf.data`, for ragged or TF-only inputs."""
    if tf is None:
        raise ImportError(
            "Preprocessing these inputs requires `grain` or `tensorflow`. "
            "Run `pip install grain` to install Grain, the pure-Python data "
            "loader used by KerasHub."
        )
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
            raise ValueError(UNBATCHED_INPUT_ERROR) from e
        raise e

    return ds.batch(batch_size or 32)


def _apply_preprocessing(ds, preprocess_samples):
    """Map `preprocess_samples` over a batched dataset, with prefetching."""
    if _is_tf_dataset(ds):
        return ds.map(
            preprocess_samples, num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)

    if not _is_grain_dataset(ds):
        raise ValueError(
            "Expected `x` to be a `grain` or `tf.data` dataset. Received: "
            f"`type(x)={type(ds)}`."
        )

    def preprocess(element):
        # Grain passes the whole element as one argument, unlike `tf.data`.
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(element)
        # Convert strings after unpacking, or `unpack_x_y_sample_weight` would
        # read the samples of a single string input as separate fields.
        x = _convert_strings_to_python(x)
        y = _convert_strings_to_python(y)
        sample_weight = _convert_strings_to_python(sample_weight)
        outputs = preprocess_samples(x, y, sample_weight)
        # `preprocess_samples` can return tf tensors, which jax rejects since
        # `GrainDatasetAdapter.get_jax_iterator` does not convert them. This
        # is the same conversion preprocessing layers apply to their own
        # outputs inside a grain pipeline, ragged and string data included.
        return convert_preprocessing_outputs_grain(outputs)

    # `MapDataset.__iter__` already calls `to_iter_dataset()`, which reads
    # ahead on a thread pool. Calling it here would only discard `__len__`.
    return ds.map(preprocess)


@contextlib.contextmanager
def _silence_unknown_length_warning(ds, steps):
    """Hide the spurious "ran out of data" warning for grain datasets.

    `GrainDatasetAdapter.num_batches` is always `None`, so Keras finds the
    epoch size by running the iterator dry. Before keras-team/keras#23360 that
    warned as if training had been cut short, once per epoch and again for
    `evaluate()` and `predict()`. The warning only carries information when
    the caller declared how many steps to expect, which is the condition
    reproduced here.

    TODO: drop this once the minimum supported Keras includes that fix, which
    is unreleased as of Keras 3.15.1.
    """
    if steps is not None or not _is_grain_dataset(ds):
        yield
        return
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Your input ran out of data",
            category=UserWarning,
        )
        yield


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
    def __call__(self, *args, **kwargs):
        inputs = args[0] if args else kwargs.get("inputs")
        if _contains_string_data(inputs):
            message = (
                "Calling a model directly, e.g. `model(x)`, does not apply "
                "preprocessing, but the model received string input. Use "
                "`model.predict(x)`, `model.fit(x, y)` or "
                "`model.evaluate(x, y)` instead, which will preprocess "
                "string input before running the model."
            )
            if getattr(self, "preprocessor", None) is not None:
                message += (
                    " Alternatively, preprocess the input first, e.g. "
                    "`model(model.preprocessor(x))`."
                )
            raise ValueError(message)
        return super().__call__(*args, **kwargs)

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
        x = _apply_preprocessing(x, self.preprocess_samples)

        if validation_data is not None:
            if _is_tf_dataset(validation_data) or _is_grain_dataset(
                validation_data
            ):
                validation_data = _apply_preprocessing(
                    validation_data, self.preprocess_samples
                )
            else:
                (vx, vy, vsw) = keras.utils.unpack_x_y_sample_weight(
                    validation_data
                )
                validation_data = _apply_preprocessing(
                    _convert_inputs_to_dataset(vx, vy, vsw, batch_size),
                    self.preprocess_samples,
                )

        with _silence_unknown_length_warning(x, kwargs.get("steps_per_epoch")):
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
        # `fit()` already preprocessed `validation_data`, so use the iterator
        # `keras.Model` cached from it rather than building another one.
        if kwargs.get("_use_cached_eval_dataset", False):
            return super().evaluate(
                x=x,
                y=y,
                batch_size=batch_size,
                sample_weight=sample_weight,
                **kwargs,
            )
        x = _convert_inputs_to_dataset(x, y, sample_weight, batch_size)
        x = _apply_preprocessing(x, self.preprocess_samples)
        with _silence_unknown_length_warning(x, kwargs.get("steps")):
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
        x = _apply_preprocessing(x, self.preprocess_samples)
        with _silence_unknown_length_warning(x, kwargs.get("steps")):
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
        x = tree.map_structure(ops.convert_to_tensor, x)
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
        x = tree.map_structure(ops.convert_to_tensor, x)
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
        x = tree.map_structure(ops.convert_to_tensor, x)
        return super().predict_on_batch(
            x=x,
            **kwargs,
        )
