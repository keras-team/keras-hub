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
"""Gemma model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "gemma_2b_en": {
        "metadata": {
            "description": (
                "18-layer Gemma model (Gemma with 2B parameters). "
            ),
            "params": 2506172416,
            "official_name": "Gemma",
            "path": "gemma",
            "model_card": "https://www.kaggle.com/models/google/gemma",
        },
        "kaggle_handle": "kaggle://keras/gemma/keras/gemma_2b_en/2",
    },
    "gemma_instruct_2b_en": {
        "metadata": {
            "description": (
                "18-layer Gemma model (Gemma with 2B parameters). "
            ),
            "params": 2506172416,
            "official_name": "Gemma",
            "path": "gemma",
            "model_card": "https://www.kaggle.com/models/google/gemma",
        },
        "kaggle_handle": "kaggle://keras/gemma/keras/gemma_instruct_2b_en/2",
    },
    "gemma_7b_en": {
        "metadata": {
            "description": (
                "28-layer Gemma model (Gemma with 7B parameters). "
            ),
            "params": 8537680896,
            "official_name": "Gemma",
            "path": "gemma",
            "model_card": "https://www.kaggle.com/models/google/gemma",
        },
        "kaggle_handle": "kaggle://keras/gemma/keras/gemma_7b_en/2",
    },
    "gemma_instruct_7b_en": {
        "metadata": {
            "description": (
                "28-layer Gemma model (Gemma with 7B parameters). "
            ),
            "params": 8537680896,
            "official_name": "Gemma",
            "path": "gemma",
            "model_card": "https://www.kaggle.com/models/google/gemma",
        },
        "kaggle_handle": "kaggle://keras/gemma/keras/gemma_instruct_7b_en/2",
    },
}
