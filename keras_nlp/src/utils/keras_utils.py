# Copyright 2023 The KerasNLP Authors
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

import sys

import keras
from absl import logging
from packaging.version import parse

from keras_nlp.src.utils.tensor_utils import is_tensor_type

try:
    import tensorflow as tf
except ImportError:
    tf = None


def clone_initializer(initializer):
    """Clones an initializer to ensure a new seed.

    As of tensorflow 2.10, we need to clone user passed initializers when
    invoking them twice to avoid creating the same randomized initialization.
    """
    # If we get a string or dict, just return as we cannot and should not clone.
    if not isinstance(initializer, keras.initializers.Initializer):
        return initializer
    config = initializer.get_config()
    return initializer.__class__.from_config(config)


def convert_inputs_to_list_of_tensor_segments(x):
    """Converts user inputs to a list of a tensor segments.

    For models and layers which accept lists of string tensors to pack together,
    this method converts user inputs to a uniform format in a way that can be
    considered canonical for the library.

    We handle the following:

    - A single string will be converted to a tensor and wrapped in a list.
    - A list of strings will be converted to a tensor and wrapped in a list.
    - A single tensor will be wrapped in a list.
    - A list of tensors will be passed through unaltered.

    All other inputs will result in an error. This effectively means that users
    who would like to pack multiple segments together should convert those
    segments to tensors before calling the layer. This removes any ambiguity
    in the input for those cases.
    """
    # Check the input type.
    is_string = isinstance(x, (str, bytes))
    is_tensor = is_tensor_type(x)
    is_string_list = (
        isinstance(x, (list, tuple)) and x and isinstance(x[0], (str, bytes))
    )
    is_tensor_list = isinstance(x, (list, tuple)) and x and is_tensor_type(x[0])

    if is_string or is_string_list:
        # Automatically convert raw strings or string lists to tensors.
        # Wrap this input as a single (possibly batched) segment.
        x = [tf.convert_to_tensor(x)]
    elif is_tensor:
        # Automatically wrap a single tensor as a single segment.
        x = [x]
    elif is_tensor_list:
        # Pass lists of tensors though unaltered.
        x = x
    else:
        # Error for all other input.
        raise ValueError(
            f"Unsupported input for `x`. `x` should be a string, a list of "
            "strings, or a list of tensors. If passing multiple segments "
            "which should packed together, please convert your inputs to a "
            f"list of tensors. Received `x={x}`"
        )
    return x


def print_msg(message, line_break=True):
    """Print the message to absl logging or stdout."""
    # Copied from core Keras.
    if keras.utils.is_interactive_logging_enabled():
        if line_break:
            sys.stdout.write(message + "\n")
        else:
            sys.stdout.write(message)
        sys.stdout.flush()
    else:
        logging.info(message)


@keras.saving.register_keras_serializable(package="keras_nlp")
def gelu_approximate(x):
    return keras.activations.gelu(x, approximate=True)


def has_quantization_support():
    return False if parse(keras.version()) < parse("3.4.0") else True


def assert_quantization_support():
    if not has_quantization_support():
        raise ValueError(
            "Quantization API requires Keras >= 3.4.0 to function "
            f"correctly. Received: '{keras.version()}'"
        )


def standardize_data_format(data_format):
    if data_format is None:
        return keras.config.image_data_format()
    data_format = str(data_format).lower()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(
            "The `data_format` argument must be one of "
            "{'channels_first', 'channels_last'}. "
            f"Received: data_format={data_format}"
        )
    return data_format


def get_feature_extractor(model, layer_names, output_keys=None):
    """Create a feature extractor model with augmented output.

    This method produces a new `keras.Model` with the same input signature
    as the source but with the layers in `layer_names` as the output.
    This is useful for downstream tasks that require more output than the
    final layer of the backbone.

    Args:
        model: keras.Model. The source model.
        layer_names: list of strings. Names of layers to include in the
            output signature.
        output_keys: optional, list of strings. Key to use for each layer in
            the model's output dictionary.

    Returns:
        `keras.Model` which has dict as outputs.
    """

    if not output_keys:
        output_keys = layer_names
    items = zip(output_keys, layer_names)
    outputs = {key: model.get_layer(name).output for key, name in items}
    return keras.Model(inputs=model.inputs, outputs=outputs)


def detect_if_tensorflow_uses_keras_3():
    # We follow the version of keras that tensorflow is configured to use.
    try:
        from tensorflow import keras

        # Note that only recent versions of keras have a `version()` function.
        if hasattr(keras, "version") and keras.version().startswith("3."):
            return True
    except:
        raise ValueError(
            "Unable to import `keras` with `tensorflow`.  Please check your "
            "Keras and Tensorflow version are compatible; Keras 3 requires "
            "TensorFlow 2.15 or later. See keras.io/getting_started for more "
            "information on installing Keras."
        )

    # No `keras.version()` means we are on an old version of keras.
    return False


_USE_KERAS_3 = detect_if_tensorflow_uses_keras_3()


def keras_3():
    """Check if Keras 3 is being used."""
    return _USE_KERAS_3


def get_tensor_input_name(tensor):
    if keras_3():
        return tensor._keras_history.operation.name
    else:
        return tensor.node.layer.name


def parse_model_inputs(input_shape, input_tensor, **kwargs):
    if input_tensor is None:
        return keras.layers.Input(shape=input_shape, **kwargs)
    else:
        if not keras.backend.is_keras_tensor(input_tensor):
            return keras.layers.Input(
                tensor=input_tensor, shape=input_shape, **kwargs
            )
        else:
            return input_tensor


def correct_pad_downsample(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    Args:
        inputs: Input tensor.
        kernel_size: An integer or tuple/list of 2 integers.

    Returns:
        A tuple.
    """
    img_dim = 1
    input_size = inputs.shape[img_dim : (img_dim + 2)]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return (
        (correct[0] - adjust[0], correct[0]),
        (correct[1] - adjust[1], correct[1]),
    )