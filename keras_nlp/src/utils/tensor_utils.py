# Copyright 2024 The KerasNLP Authors
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

import contextlib
import functools
import inspect
import threading

import keras
from keras import ops

from keras_nlp.src.api_export import keras_nlp_export

try:
    import tensorflow as tf
    import tensorflow_text as tf_text
except ImportError:
    tf = None
    tf_text = None


NO_CONVERT_COUNTER = threading.local()
NO_CONVERT_COUNTER.count = 0


@contextlib.contextmanager
def no_convert_scope():
    try:
        NO_CONVERT_COUNTER.count += 1
        yield
    finally:
        NO_CONVERT_COUNTER.count -= 1


def in_no_convert_scope():
    return NO_CONVERT_COUNTER.count > 0


def tf_preprocessing_function(fn):
    """Wraps a preprocessing function to handle tf tensor conversion."""
    if tf is None:
        return fn

    params = inspect.signature(fn).parameters
    accepts_labels = all(k in params for k in ("x", "y", "sample_weight"))
    with tf.device("cpu"):
        if not accepts_labels:

            @functools.wraps(fn)
            def wrapper(self, x, **kwargs):
                x = convert_to_tf(x)
                with no_convert_scope():
                    x = fn(self, x, **kwargs)
                return convert_from_tf(x)

        else:

            @functools.wraps(fn)
            def wrapper(self, x, y=None, sample_weight=None, **kwargs):
                x, y, sample_weight = convert_to_tf((x, y, sample_weight))
                with no_convert_scope():
                    x = fn(self, x, y=y, sample_weight=sample_weight, **kwargs)
                return convert_from_tf(x)

        return wrapper


@keras_nlp_export("keras_nlp.utils.convert_to_tf")
def convert_to_tf(x):
    """Convert raw inputs to tf inputs for preprocessing.

    This function is used to convert raw inputs (strings, lists, `np.ndarray`s,
    `jax.Array`s, `torch.Tensor`s) to tensorflow inputs for use with `tf.data`
    and KerasNLP preprocessing layers. It will convert ragged inputs and string
    inputs `tf.RaggedTensor` and `tf.Tensor` types. This will automatically be
    called when running preprocessing layers or `keras_nlp.models.Task`s with
    preprocessing included.

    `tuple` and `list` elements are handled differently by this function. A
    `tuple` is assumed to enumerate separate inputs, and a `list` is assumed to
    enumerate elements in a single array-like input. This makes it possible to
    represent ragged and string inputs in a multi-backend format, as shown in
    the examples below.

    Examples:
    ```python
    # Two ragged arrays of token ids.
    x = ([[1, 2, 3], [4, 5]], [[1, 2], [3, 4, 5]])
    keras_nlp.utils.convert_to_tf(x)

    # A batch of three samples each with two string segments.
    x = (["hi", "hello", "hey"], ["bye", "later", "so long"])
    keras_nlp.utils.convert_to_tf(x)

    # A batch of features in a dictionary.
    x = {
        "text": ["hi", "hello", "hey"],
        "images": np.ones((3, 64, 64, 3)),
        "labels": [1, 0, 1],
    }
    keras_nlp.utils.convert_to_tf(x)
    ```
    """
    if isinstance(x, dict):
        return {k: convert_to_tf(x[k]) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(convert_to_tf(v) for v in x)
    if isinstance(x, list):
        return tf.ragged.constant(x)
    if isinstance(x, str):
        return tf.constant(x)
    if is_tensor_type(x):
        # Torch will complain about device placement for GPU tensors.
        if keras.config.backend() == "torch":
            import torch

            if isinstance(x, torch.Tensor):
                x = x.cpu()
        return tf.convert_to_tensor(x)
    return x


@keras_nlp_export("keras_nlp.utils.convert_from_tf")
def convert_from_tf(x):
    """Convert tf outputs after preprocessing to a backend agnostic format.

    This function is used to convert `tf.Tensor` and `tf.RaggedTensor` output
    from preprocessing layers to either:

    - The correct tensor type for the Keras backend framework.
    - Python lists, in the case of ragged and string data.

    This will automatically be called when on the output of preprocessing
    layers or `keras_nlp.models.Task`s with preprocessing included. It could be
    used directly to convert a `tf.data.Dataset` output to a backend agnostic
    type.

    Examples:
    ```python
    # Two ragged arrays of token ids.
    x = tf.ragged.constant([[1, 2, 3], [4, 5]])
    keras_nlp.utils.convert_from_tf(x)

    # A batch of three samples each with two string segments.
    x = (tf.constant["hi", "yo", "hey"]), tf.constant(["bye", "ciao", ""]))
    keras_nlp.utils.convert_from_tf(x)

    # A batch of features in a dictionary.
    x = {
        "text": tf.constant(["hi", "hello", "hey"]),
        "images": tf.ones((3, 64, 64, 3)),
        "labels": tf.constant([1, 0, 1]),
    }
    keras_nlp.utils.convert_from_tf(x)
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
            "KerasNLP uses `tf.data` and `tensorflow-text` to preprocess text "
            "on all Keras backends. If you are running on Jax or Torch, this "
            "installation does not need GPU support."
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
        values: List or iterable of tensors shaped like `inputs` or broadcastable
            by bit operators.
        padding_mask: Tensor with shape compatible with inputs that will condition
            output.

    Returns:
        A tensor with `inputs` shape where each position is True if it contains
            a value from any `values`. Padding mask will be applied before
            returning."""
    output = ops.equal(inputs, values[0])
    for value in values[1:]:
        value_equality = ops.equal(inputs, value)
        output = ops.logical_or(output, value_equality)

    return ops.logical_and(output, padding_mask)
