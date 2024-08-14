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

from keras_nlp.src.utils.preset_utils import HF_CONFIG_FILE
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
                "Converting from the huggingface/timm model format"
                "requires the safetensors package."
                "Please install with `pip install safetensors`."
            )

        self.preset = preset
        if check_file_exists(preset, HF_CONFIG_FILE):
            self.safetensor_config = load_config(preset, HF_CONFIG_FILE)
        else:
            self.safetensor_config = None

        self.file = None

    def __enter__(self):
        path = get_file(self.preset, SAFETENSOR_FILE)
        self.file = self.enter_context(
            safetensors.safe_open(path, framework="np")
        )
        return super().__enter__()

    def get_tensor(self, hf_weight_key):
        if self.file is None:
            path = get_file(self.preset, SAFETENSOR_FILE)
            file = self.enter_context(
                safetensors.safe_open(path, framework="np")
            )
        else:
            file = self.file
        return file.get_tensor(hf_weight_key)

    def port_weight(self, keras_variable, hf_weight_key, hook_fn=None):
        hf_tensor = self.get_tensor(hf_weight_key)
        if hook_fn:
            hf_tensor = hook_fn(hf_tensor, list(keras_variable.shape))
        keras_variable.assign(hf_tensor)
