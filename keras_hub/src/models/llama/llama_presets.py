# Copyright 2024 The KerasHub Authors
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
    "llama2_7b_en": {
        "metadata": {
            "description": "7 billion parameter, 32-layer, base LLaMA 2 model.",
            "params": 6738415616,
            "official_name": "LLaMA 2",
            "path": "llama2",
            "model_card": "https://github.com/meta-llama/llama",
        },
        "kaggle_handle": "kaggle://keras/llama2/keras/llama2_7b_en/1",
    },
    "llama2_7b_en_int8": {
        "metadata": {
            "description": (
                "7 billion parameter, 32-layer, base LLaMA 2 model with "
                "activation and weights quantized to int8."
            ),
            "params": 6739839488,
            "official_name": "LLaMA 2",
            "path": "llama2",
            "model_card": "https://github.com/meta-llama/llama",
        },
        "kaggle_handle": "kaggle://keras/llama2/keras/llama2_7b_en_int8/1",
    },
    "llama2_instruct_7b_en": {
        "metadata": {
            "description": (
                "7 billion parameter, 32-layer, instruction tuned LLaMA 2 "
                "model."
            ),
            "params": 6738415616,
            "official_name": "LLaMA 2",
            "path": "llama2",
            "model_card": "https://github.com/meta-llama/llama",
        },
        "kaggle_handle": "kaggle://keras/llama2/keras/llama2_instruct_7b_en/1",
    },
    "llama2_instruct_7b_en_int8": {
        "metadata": {
            "description": (
                "7 billion parameter, 32-layer, instruction tuned LLaMA 2 "
                "model with activation and weights quantized to int8."
            ),
            "params": 6739839488,
            "official_name": "LLaMA 2",
            "path": "llama2",
            "model_card": "https://github.com/meta-llama/llama",
        },
        "kaggle_handle": "kaggle://keras/llama2/keras/llama2_instruct_7b_en_int8/1",
    },
    "vicuna_1.5_7b_en": {
        "metadata": {
            "description": (
                "7 billion parameter, 32-layer, instruction tuned Vicuna v1.5 "
                "model."
            ),
            "params": 6738415616,
            "official_name": "Vicuna",
            "path": "vicuna",
            "model_card": "https://github.com/lm-sys/FastChat",
        },
        "kaggle_handle": "kaggle://keras/vicuna/keras/vicuna_1.5_7b_en/1",
    },
}
