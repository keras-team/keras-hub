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
"""Llama model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "llama_7b_en": {
        "metadata": {
            "description": "Llama 7B Base model",
            "params": 6738415616,
            "official_name": "Llama",
            "path": "llama",
            "model_card": "https://github.com/llamaai/llama-src/blob/main/README.md",
        },
        "kaggle_handle": "kaggle://keras/llama/keras/llama_7b_en/1",
    },
    "llama_instruct_7b_en": {
        "metadata": {
            "description": "LLaMA 7B Chat model",
            "params": 6738415616,
            "official_name": "LLaMA",
            "path": "llama",
            "model_card": "https://github.com/llamaai/llama-src/blob/main/README.md",
        },
        "kaggle_handle": "kaggle://keras/llama/keras/llama_instruct_7b_en/1",
    },
}
