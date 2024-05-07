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
"""PaliGemma model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "pali_gemma_3b_mix_224": {
        "metadata": {
            "description": (
                "image size 224, mix fine tuned, text sequence " "length is 256"
            ),
            "params": 2923335408,
            "official_name": "PaliGemma",
            "path": "pali_gemma",
            "model_card": "https://www.kaggle.com/models/google/paligemma",
        },
        "kaggle_handle": "kaggle://keras/paligemma/keras/pali_gemma_3b_mix_224/1",
    },
    "pali_gemma_3b_mix_448": {
        "metadata": {
            "description": (
                "image size 448, mix fine tuned, text sequence length is 512"
            ),
            "params": 2924220144,
            "official_name": "PaliGemma",
            "path": "pali_gemma",
            "model_card": "https://www.kaggle.com/models/google/paligemma",
        },
        "kaggle_handle": "kaggle://keras/paligemma/keras/pali_gemma_3b_mix_448/1",
    },
    "pali_gemma_3b_224": {
        "metadata": {
            "description": (
                "image size 224, pre trained, text sequence length is 128"
            ),
            "params": 2923335408,
            "official_name": "PaliGemma",
            "path": "pali_gemma",
            "model_card": "https://www.kaggle.com/models/google/paligemma",
        },
        "kaggle_handle": "kaggle://keras/paligemma/keras/pali_gemma_3b_224/1",
    },
    "pali_gemma_3b_448": {
        "metadata": {
            "description": (
                "image size 448, pre trained, text sequence length is 512"
            ),
            "params": 2924220144,
            "official_name": "PaliGemma",
            "path": "pali_gemma",
            "model_card": "https://www.kaggle.com/models/google/paligemma",
        },
        "kaggle_handle": "kaggle://keras/paligemma/keras/pali_gemma_3b_448/1",
    },
    "pali_gemma_3b_896": {
        "metadata": {
            "description": (
                "image size 896, pre trained, text sequence length " "is 512"
            ),
            "params": 2927759088,
            "official_name": "PaliGemma",
            "path": "pali_gemma",
            "model_card": "https://www.kaggle.com/models/google/paligemma",
        },
        "kaggle_handle": "kaggle://keras/paligemma/keras/pali_gemma_3b_896/1",
    },
}
