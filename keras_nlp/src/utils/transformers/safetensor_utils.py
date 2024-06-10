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
import einops
try: 
    import safetensors 
except ImportError: 
    safetensors = None


def set_keras_weights(
    safetensor_files,
    safetensor_config,
    keras_layer,
    hf_weight_keys,
    rearrange_patterns=None,
    rearrange_dims=None,
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
    if rearrange_patterns and isinstance(rearrange_patterns, str):
        rearrange_patterns = [rearrange_patterns] * len(hf_weight_keys)
    elif not rearrange_patterns:
        rearrange_patterns = [None] * len(hf_weight_keys)

    tensors = []
    for hf_weight_key, rearrange_pattern in zip(
        hf_weight_keys, rearrange_patterns
    ):
        safetensor_file = safetensor_files[
            safetensor_config["weight_map"][hf_weight_key]
        ]
        with safe_open(safetensor_file, framework="np") as f:
            tensor = f.get_tensor(hf_weight_key)
            if rearrange_pattern:
                tensor = einops.rearrange(
                    tensor,
                    rearrange_pattern,
                    **rearrange_dims if rearrange_dims else {}
                )
            tensors.append(tensor)
    keras_layer.set_weights(tensors)
