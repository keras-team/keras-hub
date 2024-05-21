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
"""Llama 3 model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "llama3_8b_en": {
        "metadata": {
            "description": "LLaMA 3 8B Base model",
            "params": 8030261248,
            "official_name": "LLaMA 3",
            "path": "llama3",
            "model_card": "https://github.com/meta-llama/llama3",
        },
        "kaggle_handle": "kaggle://keras/llama3/keras/llama3_8b_en/3",
    },
    "llama3_instruct_8b_en": {
        "metadata": {
            "description": "LLaMA 3 8B Instruct model",
            "params": 8030261248,
            "official_name": "LLaMA 3",
            "path": "llama3",
            "model_card": "https://github.com/meta-llama/llama3",
        },
        "kaggle_handle": "kaggle://keras/llama3/keras/llama3_instruct_8b_en/3",
    },
}
