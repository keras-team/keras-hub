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

# Metadata for loading pretrained model weights.
backbone_presets = {
    "whisper_tiny_en": {
        "metadata": {
            "description": (
                "4-layer Whisper model. Trained on 438,000 hours of labelled "
                "English speech data."
            ),
            "params": 37184256,
            "official_name": "Whisper",
            "path": "whisper",
            "model_card": "https://github.com/openai/whisper/blob/main/model-card.md",
        },
        "kaggle_handle": "kaggle://keras/whisper/keras/whisper_tiny_en/2",
    },
    "whisper_base_en": {
        "metadata": {
            "description": (
                "6-layer Whisper model. Trained on 438,000 hours of labelled "
                "English speech data."
            ),
            "params": 124439808,
            "official_name": "Whisper",
            "path": "whisper",
            "model_card": "https://github.com/openai/whisper/blob/main/model-card.md",
        },
        "kaggle_handle": "kaggle://keras/whisper/keras/whisper_base_en/2",
    },
    "whisper_small_en": {
        "metadata": {
            "description": (
                "12-layer Whisper model. Trained on 438,000 hours of labelled "
                "English speech data."
            ),
            "params": 241734144,
            "official_name": "Whisper",
            "path": "whisper",
            "model_card": "https://github.com/openai/whisper/blob/main/model-card.md",
        },
        "kaggle_handle": "kaggle://keras/whisper/keras/whisper_small_en/2",
    },
    "whisper_medium_en": {
        "metadata": {
            "description": (
                "24-layer Whisper model. Trained on 438,000 hours of labelled "
                "English speech data."
            ),
            "params": 763856896,
            "official_name": "Whisper",
            "path": "whisper",
            "model_card": "https://github.com/openai/whisper/blob/main/model-card.md",
        },
        "kaggle_handle": "kaggle://keras/whisper/keras/whisper_medium_en/2",
    },
    "whisper_tiny_multi": {
        "metadata": {
            "description": (
                "4-layer Whisper model. Trained on 680,000 hours of labelled "
                "multilingual speech data."
            ),
            "params": 37760640,
            "official_name": "Whisper",
            "path": "whisper",
            "model_card": "https://github.com/openai/whisper/blob/main/model-card.md",
        },
        "kaggle_handle": "kaggle://keras/whisper/keras/whisper_tiny_multi/2",
    },
    "whisper_base_multi": {
        "metadata": {
            "description": (
                "6-layer Whisper model. Trained on 680,000 hours of labelled "
                "multilingual speech data."
            ),
            "params": 72593920,
            "official_name": "Whisper",
            "path": "whisper",
            "model_card": "https://github.com/openai/whisper/blob/main/model-card.md",
        },
        "kaggle_handle": "kaggle://keras/whisper/keras/whisper_base_multi/2",
    },
    "whisper_small_multi": {
        "metadata": {
            "description": (
                "12-layer Whisper model. Trained on 680,000 hours of labelled "
                "multilingual speech data."
            ),
            "params": 241734912,
            "official_name": "Whisper",
            "path": "whisper",
            "model_card": "https://github.com/openai/whisper/blob/main/model-card.md",
        },
        "kaggle_handle": "kaggle://keras/whisper/keras/whisper_small_multi/2",
    },
    "whisper_medium_multi": {
        "metadata": {
            "description": (
                "24-layer Whisper model. Trained on 680,000 hours of labelled "
                "multilingual speech data."
            ),
            "params": 763857920,
            "official_name": "Whisper",
            "path": "whisper",
            "model_card": "https://github.com/openai/whisper/blob/main/model-card.md",
        },
        "kaggle_handle": "kaggle://keras/whisper/keras/whisper_medium_multi/2",
    },
    "whisper_large_multi": {
        "metadata": {
            "description": (
                "32-layer Whisper model. Trained on 680,000 hours of labelled "
                "multilingual speech data."
            ),
            "params": 1543304960,
            "official_name": "Whisper",
            "path": "whisper",
            "model_card": "https://github.com/openai/whisper/blob/main/model-card.md",
        },
        "kaggle_handle": "kaggle://keras/whisper/keras/whisper_large_multi/2",
    },
    "whisper_large_multi_v2": {
        "metadata": {
            "description": (
                "32-layer Whisper model. Trained for 2.5 epochs on 680,000  "
                "hours of labelled multilingual speech data. An improved "
                "of `whisper_large_multi`."
            ),
            "params": 1543304960,
            "official_name": "Whisper",
            "path": "whisper",
            "model_card": "https://github.com/openai/whisper/blob/main/model-card.md",
        },
        "kaggle_handle": "kaggle://keras/whisper/keras/whisper_large_multi_v2/2",
    },
}
