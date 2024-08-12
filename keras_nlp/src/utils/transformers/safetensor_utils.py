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

from keras_nlp.src.utils.preset_utils import SAFETENSOR_CONFIG_FILE
from keras_nlp.src.utils.preset_utils import SAFETENSOR_FILE
from keras_nlp.src.utils.preset_utils import check_file_exists
from keras_nlp.src.utils.preset_utils import get_file
from keras_nlp.src.utils.preset_utils import load_config

try:
    import safetensors
except ImportError:
    safetensors = None


class SafetensorLoader(contextlib.ExitStack):
    def __init__(self, preset):
        super().__init__()

        if safetensors is None:
            raise ImportError(
                "Converting from the huggingface/transformers model format"
                "requires the safetensors package."
                "Please install with `pip install safetensors`."
            )

        self.preset = preset
        if check_file_exists(preset, SAFETENSOR_CONFIG_FILE):
            self.safetensor_config = load_config(preset, SAFETENSOR_CONFIG_FILE)
        else:
            self.safetensor_config = None
        self.safetensor_files = {}
        self.prefix = None

    def get_prefixed_key(self, hf_weight_key, dict_like):
        """
        Determine and return a prefixed key for a given hf weight key.

        This method checks if there's a common prefix for the weight keys and caches it
        for future use.

        Args:
            hf_weight_key (str): The hf weight key to check for a prefix.
            dict_like (object): An object to get keys of safetensor file using keys() method.

        Returns:
            str: The full key including the prefix (if any).
        """
        if self.prefix is not None:
            return self.prefix + hf_weight_key

        for full_key in dict_like.keys():
            if full_key.endswith(hf_weight_key) and full_key != hf_weight_key:
                self.prefix = full_key[: -len(hf_weight_key)]
                return full_key

        self.prefix = ""
        return hf_weight_key

    def get_tensor(self, hf_weight_key):
        if self.safetensor_config is None:
            fname = SAFETENSOR_FILE
        else:
            full_key = self.get_prefixed_key(
                hf_weight_key, self.safetensor_config["weight_map"]
            )
            fname = self.safetensor_config["weight_map"][full_key]

        if fname in self.safetensor_files:
            file = self.safetensor_files[fname]
        else:
            path = get_file(self.preset, fname)
            file = self.enter_context(
                safetensors.safe_open(path, framework="np")
            )
            self.safetensor_files[fname] = file

        full_key = self.get_prefixed_key(hf_weight_key, file)
        return file.get_tensor(full_key)

    def port_weight(self, keras_variable, hf_weight_key, hook_fn=None):
        hf_tensor = self.get_tensor(hf_weight_key)
        if hook_fn:
            hf_tensor = hook_fn(hf_tensor, list(keras_variable.shape))
        keras_variable.assign(hf_tensor)
