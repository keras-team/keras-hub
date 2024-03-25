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
"""Mistral model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "mistral_7b_en": {
        "metadata": {
            "description": "Mistral 7B base model",
            "params": 7241732096,
            "official_name": "Mistral",
            "path": "mistral",
            "model_card": "https://github.com/mistralai/mistral-src/blob/main/README.md",
        },
        "kaggle_handle": "kaggle://keras/mistral/keras/mistral_7b_en/6",
    },
    "mistral_instruct_7b_en": {
        "metadata": {
            "description": "Mistral 7B instruct model",
            "params": 7241732096,
            "official_name": "Mistral",
            "path": "mistral",
            "model_card": "https://github.com/mistralai/mistral-src/blob/main/README.md",
        },
        "kaggle_handle": "kaggle://keras/mistral/keras/mistral_instruct_7b_en/6",
    },
    "mistral_0.2_instruct_7b_en": {
        "metadata": {
            "description": "Mistral 7B instruct Version 0.2 model",
            "params": 7241732096,
            "official_name": "Mistral",
            "path": "mistral",
            "model_card": "https://github.com/mistralai/mistral-src/blob/main/README.md",
        },
        "kaggle_handle": "kaggle://keras/mistral/keras/mistral_0.2_instruct_7b_en/1",
    },
}
