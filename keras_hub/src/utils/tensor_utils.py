import contextlib
import functools
import inspect
import threading

import keras
import numpy as np
from keras import ops
from packaging import version

try:
    import tensorflow as tf
    import tensorflow_text as tf_text
except ImportError:
    tf = None
    tf_text = None


NO_CONVERT_COUNTER = threading.local()


@contextlib.contextmanager
def no_convert_scope():
    try:
        NO_CONVERT_COUNTER.count = getattr(NO_CONVERT_COUNTER, "count", 0) + 1
        yield
    finally:
        NO_CONVERT_COUNTER.count = getattr(NO_CONVERT_COUNTER, "count", 0) - 1


def in_no_convert_scope():
    return getattr(NO_CONVERT_COUNTER, "count", 0) > 0


def preprocessing_function(fn):
    """Wraps a preprocessing function to handle tf tensor conversion."""
    if tf is None:
        return fn

    params = inspect.signature(fn).parameters
    accepts_labels = all(k in params for k in ("x", "y", "sample_weight"))
    if not accepts_labels:

        @functools.wraps(fn)
        def wrapper(self, x, **kwargs):
            with tf.device("cpu"):
                x = convert_preprocessing_inputs(x)
                with no_convert_scope():
                    x = fn(self, x, **kwargs)
                return convert_preprocessing_outputs(x)

    else:

        @functools.wraps(fn)
        def wrapper(self, x, y=None, sample_weight=None, **kwargs):
            with tf.device("cpu"):
                x, y, sample_weight = convert_preprocessing_inputs(
                    (x, y, sample_weight)
                )
                with no_convert_scope():
                    x = fn(self, x, y=y, sample_weight=sample_weight, **kwargs)
                return convert_preprocessing_outputs(x)

    return wrapper


def convert_preprocessing_inputs(x):
    """Convert raw inputs for preprocessing.

    This function is used to convert raw inputs (strings, lists, `np.ndarray`s,
    `jax.Array`s, `torch.Tensor`s, etc) to a canonical format for
    preprocessing layers. All inputs will be converted to backend tensors if
    possible, except ragged inputs and string inputs which be converted to tf
    tensors regardless of backend.

    `tuple` and `list` elements are handled differently by this function. A
    `tuple` is assumed to enumerate separate inputs, and a `list` is assumed to
    enumerate elements in a single array-like input. This makes it possible to
    represent ragged and string inputs in a multi-backend format, as shown in
    the examples below.

    Examples:
    ```python
    # Two ragged arrays of token ids.
    x = ([[1, 2, 3], [4, 5]], [[1, 2], [3, 4, 5]])
    keras_hub.utils.convert_preprocessing_inputs(x)

    # A batch of three samples each with two string segments.
    x = (["hi", "hello", "hey"], ["bye", "later", "so long"])
    keras_hub.utils.convert_preprocessing_inputs(x)

    # A batch of features in a dictionary.
    x = {
        "text": ["hi", "hello", "hey"],
        "images": np.ones((3, 64, 64, 3)),
        "labels": [1, 0, 1],
    }
    keras_hub.utils.convert_preprocessing_inputs(x)
    ```
    """
    if not tf.executing_eagerly() or in_no_convert_scope():
        return x

    if isinstance(x, dict):
        return {k: convert_preprocessing_inputs(x[k]) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(convert_preprocessing_inputs(v) for v in x)
    if isinstance(x, (str, bytes)):
        return tf.constant(x)
    if isinstance(x, list):
        try:
            numpy_x = np.array(x)
        except ValueError as e:
            # If numpy conversion failed, try converting to a ragged array.
            try:
                return tf.ragged.constant(x)
            except ValueError:
                # If ragged conversion failed return to the numpy error.
                raise e
        # If we have a string input, use tf.tensor.
        if numpy_x.dtype.type is np.str_ or numpy_x.dtype.type is np.bytes_:
            return tf.convert_to_tensor(x)
        # Numpy will default to int64, int32 works with more ops.
        if numpy_x.dtype == np.int64:
            numpy_x = numpy_x.astype(np.int32)
        # We have non-ragged, non-string input. Use backbend type.
        x = ops.convert_to_tensor(numpy_x)
        # Torch will complain about device placement for GPU tensors.
        if keras.config.backend() == "torch":
            x = x.cpu()
        return x
    if is_tensor_type(x):
        # String or ragged types we keep as tf.
        if isinstance(x, tf.RaggedTensor) or x.dtype == tf.string:
            return x
        # If we have a string input, use tf.tensor.
        if isinstance(x, np.ndarray) and x.dtype.type is np.str_:
            return tf.convert_to_tensor(x)
        x = ops.convert_to_tensor(x)
        # Torch will complain about device placement for GPU tensors.
        if keras.config.backend() == "torch":
            x = x.cpu()
        return x
    return x


def convert_preprocessing_outputs(x):
    """Convert outputs after preprocessing to a backend agnostic format.

    This function is used to convert `tf.Tensor` and `tf.RaggedTensor` output
    from preprocessing layers to either:

    - The correct tensor type for the Keras backend framework.
    - Python lists, in the case of ragged and string data.

    This will automatically be called when on the output of preprocessing
    layers or `keras_hub.models.Task`s with preprocessing included. It could be
    used directly to convert a `tf.data.Dataset` output to a backend agnostic
    type.

    Examples:
    ```python
    # Two ragged arrays of token ids.
    x = tf.ragged.constant([[1, 2, 3], [4, 5]])
    keras_hub.utils.convert_preprocessing_outputs(x)

    # A batch of three samples each with two string segments.
    x = (tf.constant["hi", "yo", "hey"]), tf.constant(["bye", "ciao", ""]))
    keras_hub.utils.convert_preprocessing_outputs(x)

    # A batch of features in a dictionary.
    x = {
        "text": tf.constant(["hi", "hello", "hey"]),
        "images": tf.ones((3, 64, 64, 3)),
        "labels": tf.constant([1, 0, 1]),
    }
    keras_hub.utils.convert_preprocessing_outputs(x)
    ```
    """
    if not tf.executing_eagerly() or in_no_convert_scope():
        return x

    def convert(x):
        if x is None:
            return x
        if isinstance(x, tf.RaggedTensor) or x.dtype == tf.string:
            return tensor_to_list(x)
        dtype = keras.backend.standardize_dtype(x.dtype)
        return ops.convert_to_tensor(x, dtype=dtype)

    return keras.tree.map_structure(convert, x)


def _decode_strings_to_utf8(inputs):
    """Recursively decodes to list of strings with 'utf-8' encoding."""
    if isinstance(inputs, bytes):
        # Handles the case when the input is a scalar string.
        return inputs.decode("utf-8", errors="ignore")
    else:
        # Recursively iterate when input is a list.
        return [_decode_strings_to_utf8(x) for x in inputs]


def tensor_to_list(inputs):
    """Converts a tensor to nested lists.

    Args:
        inputs: Input tensor, or dict/list/tuple of input tensors.
    """
    if not isinstance(inputs, (tf.RaggedTensor, tf.Tensor)):
        inputs = tf.convert_to_tensor(inputs)
    if isinstance(inputs, tf.RaggedTensor):
        list_outputs = inputs.to_list()
    elif isinstance(inputs, tf.Tensor):
        list_outputs = inputs.numpy()
        if inputs.shape.rank != 0:
            list_outputs = list_outputs.tolist()
    if inputs.dtype == tf.string:
        list_outputs = _decode_strings_to_utf8(list_outputs)
    return list_outputs


def convert_to_ragged_batch(inputs):
    """Ensure a tf.Tensor is a ragged rank 2 tensor."""
    if not isinstance(inputs, (tf.RaggedTensor, tf.Tensor)):
        inputs = tf.convert_to_tensor(inputs)
    unbatched = inputs.shape.rank == 1
    rectangular = isinstance(inputs, tf.Tensor)
    if unbatched:
        inputs = tf.expand_dims(inputs, 0)
    if rectangular:
        inputs = tf.RaggedTensor.from_tensor(inputs)
    return inputs, unbatched, rectangular


def truncate_at_token(inputs, token, mask):
    """Truncate at first instance of `token`, ignoring `mask`."""
    matches = (inputs == token) & (~mask)
    end_indices = tf.cast(tf.math.argmax(matches, -1), "int32")
    end_indices = tf.where(end_indices == 0, tf.shape(inputs)[-1], end_indices)
    return tf.RaggedTensor.from_tensor(inputs, end_indices)


def strip_to_ragged(token_ids, mask, ids_to_strip):
    """Remove masked and special tokens from a sequence before detokenizing."""
    mask = tf.cast(mask, "bool")
    for id in ids_to_strip:
        mask = mask & (token_ids != id)
    return tf.ragged.boolean_mask(token_ids, mask)


def assert_tf_libs_installed(symbol_name):
    if tf_text is None or tf is None:
        raise ImportError(
            f"{symbol_name} requires `tensorflow` and `tensorflow-text` for "
            "text processing. Run `pip install tensorflow-text` to install "
            "both packages or visit https://www.tensorflow.org/install\n\n"
            "If `tensorflow-text` is already installed, try importing it "
            "in a clean python session. Your installation may have errors.\n\n"
            "KerasHub uses `tf.data` and `tensorflow-text` to preprocess text "
            "on all Keras backends. If you are running on Jax or Torch, this "
            "installation does not need GPU support."
        )


def check_bounding_box_support():
    return version.parse(keras.__version__) >= version.parse("3.8.0")


def assert_bounding_box_support(symbol_name):
    if not check_bounding_box_support():
        raise ImportError(
            f"{symbol_name} requires Keras version to be 3.8.0 or higher. "
            f"Current keras version: {keras.__version__}"
        )


def assert_tf_backend(symbol_name):
    if keras.config.backend() != "tensorflow":
        raise RuntimeError(
            f"{symbol_name} requires the `tensorflow` backend. "
            "Please set `KERAS_BACKEND=tensorflow` when running your program."
        )


def is_tensor_type(x):
    return hasattr(x, "__array__")


def is_float_dtype(dtype):
    return "float" in keras.backend.standardize_dtype(dtype)


def is_int_dtype(dtype):
    return "int" in keras.backend.standardize_dtype(dtype)


def is_string_dtype(dtype):
    return "string" in keras.backend.standardize_dtype(dtype)


def any_equal(inputs, values, padding_mask):
    """Return a mask that is True anywhere `inputs` has a value in `values`.

    Final mask has `padding_mask` applied.

    Args:
        inputs: Input tensor.
        values: List or iterable of tensors shaped like `inputs` or
            broadcastable by bit operators.
        padding_mask: Tensor with shape compatible with inputs that will
            condition output.

    Returns:
        A tensor with `inputs` shape where each position is True if it contains
            a value from any `values`. Padding mask will be applied before
            returning."""
    output = ops.equal(inputs, values[0])
    for value in values[1:]:
        value_equality = ops.equal(inputs, value)
        output = ops.logical_or(output, value_equality)

    return ops.logical_and(output, padding_mask)


def target_gather(
    targets,
    indices,
    mask=None,
    mask_val=0.0,
):
    """A utility function wrapping `ops.take`, which deals with:
        1) both batched and unbatched `targets`.
        2) when unbatched `targets` have empty rows, the result will be filled
            with `mask_val`.
        3) target masking.

    Args:
        targets: `[N, ...]` or `[batch_size, N, ...]` Tensor representing
            targets such as boxes, keypoints, etc.
        indices: [M] or [batch_size, M] int32 Tensor representing indices within
            `targets` to gather.
        mask: `[M, ...]` or `[batch_size, M, ...]` boolean Tensor
            representing the masking for each target. `True` means the
            corresponding entity should be masked to `mask_val`, `False`
            means the corresponding entity should be the target value.
            Defaults to `None`.
        mask_val: float. representing the masking value if `mask` is True
            on the entity.
            Defaults to `0.0`

    Returns:
        targets: `[M, ...]` or `[batch_size, M, ...]` Tensor representing
            selected targets.

        Raise:
            ValueError: If `targets` is higher than rank 3.
    """
    targets_shape = list(targets.shape)
    if len(targets_shape) > 3:
        raise ValueError(
            f"`target_gather` does not support `targets` with rank "
            f"larger than 3, got {len(targets.shape)}"
        )

    def gather_unbatched(labels, match_indices, mask, mask_val):
        """Gather based on unbatched labels and boxes."""
        num_gt_boxes = labels.shape[0]

        def assign_when_rows_empty():
            if len(labels.shape) > 1:
                mask_shape = [match_indices.shape[0], labels.shape[-1]]
            else:
                mask_shape = [match_indices.shape[0]]
            return ops.cast(mask_val, labels.dtype) * ops.ones(
                mask_shape, dtype=labels.dtype
            )

        def assign_when_rows_not_empty():
            targets = ops.take(labels, match_indices, axis=0)
            if mask is None:
                return targets
            else:
                masked_targets = ops.cast(
                    mask_val, labels.dtype
                ) * ops.ones_like(mask, dtype=labels.dtype)
                return ops.where(mask, masked_targets, targets)

        if num_gt_boxes > 0:
            return assign_when_rows_not_empty()
        else:
            return assign_when_rows_empty()

    def _gather_batched(labels, match_indices, mask, mask_val):
        """Gather based on batched labels."""
        batch_size = labels.shape[0]
        if batch_size == 1:
            if mask is not None:
                result = gather_unbatched(
                    ops.squeeze(labels, axis=0),
                    ops.squeeze(match_indices, axis=0),
                    ops.squeeze(mask, axis=0),
                    mask_val,
                )
            else:
                result = gather_unbatched(
                    ops.squeeze(labels, axis=0),
                    ops.squeeze(match_indices, axis=0),
                    None,
                    mask_val,
                )
            return ops.expand_dims(result, axis=0)
        else:
            targets = ops.take_along_axis(
                labels, ops.expand_dims(match_indices, axis=-1), axis=1
            )

            if mask is None:
                return targets
            else:
                masked_targets = ops.cast(
                    mask_val, labels.dtype
                ) * ops.ones_like(mask, dtype=labels.dtype)
                return ops.where(mask, masked_targets, targets)

    if len(targets_shape) <= 2:
        return gather_unbatched(targets, indices, mask, mask_val)
    elif len(targets_shape) == 3:
        return _gather_batched(targets, indices, mask, mask_val)
