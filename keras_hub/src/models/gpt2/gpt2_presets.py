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
"""GPT-2 model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "gpt2_base_en": {
        "metadata": {
            "description": (
                "12-layer GPT-2 model where case is maintained. "
                "Trained on WebText."
            ),
            "params": 124439808,
            "official_name": "GPT-2",
            "path": "gpt2",
            "model_card": "https://github.com/openai/gpt-2/blob/master/model_card.md",
        },
        "kaggle_handle": "kaggle://keras/gpt2/keras/gpt2_base_en/2",
    },
    "gpt2_medium_en": {
        "metadata": {
            "description": (
                "24-layer GPT-2 model where case is maintained. "
                "Trained on WebText."
            ),
            "params": 354823168,
            "official_name": "GPT-2",
            "path": "gpt2",
            "model_card": "https://github.com/openai/gpt-2/blob/master/model_card.md",
        },
        "kaggle_handle": "kaggle://keras/gpt2/keras/gpt2_medium_en/2",
    },
    "gpt2_large_en": {
        "metadata": {
            "description": (
                "36-layer GPT-2 model where case is maintained. "
                "Trained on WebText."
            ),
            "params": 774030080,
            "official_name": "GPT-2",
            "path": "gpt2",
            "model_card": "https://github.com/openai/gpt-2/blob/master/model_card.md",
        },
        "kaggle_handle": "kaggle://keras/gpt2/keras/gpt2_large_en/2",
    },
    "gpt2_extra_large_en": {
        "metadata": {
            "description": (
                "48-layer GPT-2 model where case is maintained. "
                "Trained on WebText."
            ),
            "params": 1557611200,
            "official_name": "GPT-2",
            "path": "gpt2",
            "model_card": "https://github.com/openai/gpt-2/blob/master/model_card.md",
        },
        "kaggle_handle": "kaggle://keras/gpt2/keras/gpt2_extra_large_en/2",
    },
    "gpt2_base_en_cnn_dailymail": {
        "metadata": {
            "description": (
                "12-layer GPT-2 model where case is maintained. "
                "Finetuned on the CNN/DailyMail summarization dataset."
            ),
            "params": 124439808,
            "official_name": "GPT-2",
            "path": "gpt2",
        },
        "kaggle_handle": "kaggle://keras/gpt2/keras/gpt2_base_en_cnn_dailymail/2",
    },
}
