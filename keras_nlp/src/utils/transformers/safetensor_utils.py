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
try:
    import safetensors
except ImportError:
    safetensors = None


def set_keras_weight(
    safetensor_files,
    safetensor_config,
    keras_variable,
    hf_weight_key,
    hook_fn=None,
):
    if safetensors is None:
        raise ImportError(
            "Converting from the huggingface/transformers model format"
            "requires the safetensors package."
            "Please install with `pip install safetensors`."
        )
    else:
        from safetensors import safe_open

    safetensor_file = safetensor_files[
        safetensor_config["weight_map"][hf_weight_key]
    ]
    with safe_open(safetensor_file, framework="np") as f:
        hf_tensor = f.get_tensor(hf_weight_key)

        if hook_fn:
            hf_tensor = hook_fn(hf_tensor, list(keras_variable.shape))
        keras_variable.assign(hf_tensor)
