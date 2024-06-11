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
import numpy as np

try:
    import safetensors
except ImportError:
    safetensors = None


def reshape_with_pattern(tensor, pattern):
    """
    Reshapes a given tensor based on a specified pattern.

    Parameters:
    tensor (numpy.ndarray): The input tensor to reshape.
    pattern (list): A list specifying the new shape. Each element in the list can be:
        - An integer, which maps directly to the corresponding dimension of the original shape.
        - A string in the format "integer" which specifies a fixed size for the corresponding dimension.
        - A string in the format "original_index_divisor" which specifies that the corresponding dimension
          should be the size of the original dimension at 'original_index' divided by 'divisor'.

    Returns:
    numpy.ndarray: The reshaped tensor.

    Raises:
    ValueError: If the pattern format is invalid.
    TypeError: If a pattern value is not an int or str.

    Example:
    >>> tensor = np.ones((64, 128))
    >>> reshaped_tensor = reshape_with_pattern(tensor, ["8", "0_8", 1])
    >>> reshaped_tensor.shape
    (8, 8, 128)
    """
    original_shape = list(tensor.shape)
    new_shape = []

    for rule in pattern:
        if isinstance(rule, int):
            # Directly map the original shape dimension
            new_shape.append(original_shape[rule])
        elif isinstance(rule, str):
            split_rule = rule.split("_")
            if len(split_rule) == 1:
                # Use the provided integer as the size
                new_shape.append(int(split_rule[0]))
            elif len(split_rule) == 2:
                # Divide the specified original shape dimension
                original_index = int(split_rule[0])
                divisor = int(split_rule[1])
                new_shape.append(original_shape[original_index] // divisor)
            else:
                raise ValueError("Invalid pattern format")
        else:
            raise TypeError("Pattern value must be an int or str")
    return np.reshape(tensor, new_shape)


def set_keras_weights(
    safetensor_files,
    safetensor_config,
    keras_layer,
    hf_weight_keys,
    reshape_patterns=None,
    transpose_patterns=None,
):
    """
    Set Keras model weights from SafeTensors file.

    Args:
        safetensor_files (dict): Dictionary of SafeTensor file paths.
        safetensor_config (dict): Configuration for SafeTensors.
        keras_layer (keras.layers.Layer): Keras layer to set the weights for.
        hf_weight_keys (str or list): Key(s) for the Hugging Face weight(s).
        rearrange_patterns (str or list, optional): Pattern(s) for rearranging dimensions using einops.
        rearrange_dims (dict, optional): Dimensions for rearranging using einops.
    """
    if safetensors is None:
        raise ImportError(
            "`set_keras_weights()` requires the `safetensors` package. "
            "Please install with `pip install safetensors`."
        )
    else:
        from safetensors import safe_open

    if isinstance(hf_weight_keys, str):
        hf_weight_keys = [hf_weight_keys]
        reshape_patterns = [reshape_patterns] if reshape_patterns else [None]
        reshape_patterns = (
            [transpose_patterns] if transpose_patterns else [None]
        )

    tensors = []
    for hf_weight_key, reshape_pattern, transpose_pattern in zip(
        hf_weight_keys,
        reshape_patterns,
        transpose_patterns,
    ):
        safetensor_file = safetensor_files[
            safetensor_config["weight_map"][hf_weight_key]
        ]
        with safe_open(safetensor_file, framework="np") as f:
            tensor = f.get_tensor(hf_weight_key)
            if reshape_pattern:
                tensor = reshape_with_pattern(tensor, reshape_pattern)
            if transpose_pattern:
                tensor = np.transpose(tensor, axes=transpose_pattern)
            tensors.append(tensor)
    keras_layer.set_weights(tensors)
