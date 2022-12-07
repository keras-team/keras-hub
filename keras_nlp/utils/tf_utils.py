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

import numpy as np
import tensorflow as tf

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import tensorflow_text
except ImportError:
    tensorflow_text = None


def _decode_strings_to_utf8(inputs):
    """Recursively decodes to list of strings with 'utf-8' encoding."""
    if isinstance(inputs, bytes):
        # Handles the case when the input is a scalar string.
        return inputs.decode("utf-8")
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
    return list_outputs


def tensor_to_string_list(inputs):
    """Detokenize and convert tensor to nested lists of python strings.

    This is a convenience method which converts each byte string to a python
    string.

    Args:
        inputs: Input tensor, or dict/list/tuple of input tensors.
    """
    list_outputs = tensor_to_list(inputs)
    return _decode_strings_to_utf8(list_outputs)


def assert_tf_text_installed(symbol_name):
    """Detokenize and convert tensor to nested lists of python strings."""
    if tensorflow_text is None:
        raise ImportError(
            f"{symbol_name} requires the `tensorflow-text` package. "
            "Please install with `pip install tensorflow-text`."
        )


def is_tensor_type(x):
    if pd is None:
        return isinstance(x, (tf.Tensor, np.ndarray))
    else:
        return isinstance(x, (tf.Tensor, np.ndarray, pd.Series, pd.DataFrame))
