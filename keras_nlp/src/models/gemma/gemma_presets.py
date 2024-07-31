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
            "description": "2 billion parameter, 18-layer, base Gemma model.",
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
                "2 billion parameter, 18-layer, instruction tuned Gemma model."
            ),
            "params": 2506172416,
            "official_name": "Gemma",
            "path": "gemma",
            "model_card": "https://www.kaggle.com/models/google/gemma",
        },
        "kaggle_handle": "kaggle://keras/gemma/keras/gemma_instruct_2b_en/2",
    },
    "gemma_1.1_instruct_2b_en": {
        "metadata": {
            "description": (
                "2 billion parameter, 18-layer, instruction tuned Gemma model. "
                "The 1.1 update improves model quality."
            ),
            "params": 2506172416,
            "official_name": "Gemma",
            "path": "gemma",
            "model_card": "https://www.kaggle.com/models/google/gemma",
        },
        "kaggle_handle": "kaggle://keras/gemma/keras/gemma_1.1_instruct_2b_en/3",
    },
    "code_gemma_1.1_2b_en": {
        "metadata": {
            "description": (
                "2 billion parameter, 18-layer, CodeGemma model. This model "
                "has been trained on a fill-in-the-middle (FIM) task for code "
                "completion. The 1.1 update improves model quality."
            ),
            "params": 2506172416,
            "official_name": "Gemma",
            "path": "gemma",
            "model_card": "https://www.kaggle.com/models/google/gemma",
        },
        "kaggle_handle": "kaggle://keras/codegemma/keras/code_gemma_1.1_2b_en/1",
    },
    "code_gemma_2b_en": {
        "metadata": {
            "description": (
                "2 billion parameter, 18-layer, CodeGemma model. This model "
                "has been trained on a fill-in-the-middle (FIM) task for code "
                "completion."
            ),
            "params": 2506172416,
            "official_name": "Gemma",
            "path": "gemma",
            "model_card": "https://www.kaggle.com/models/google/gemma",
        },
        "kaggle_handle": "kaggle://keras/codegemma/keras/code_gemma_2b_en/1",
    },
    "gemma_7b_en": {
        "metadata": {
            "description": "7 billion parameter, 28-layer, base Gemma model.",
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
                "7 billion parameter, 28-layer, instruction tuned Gemma model."
            ),
            "params": 8537680896,
            "official_name": "Gemma",
            "path": "gemma",
            "model_card": "https://www.kaggle.com/models/google/gemma",
        },
        "kaggle_handle": "kaggle://keras/gemma/keras/gemma_instruct_7b_en/2",
    },
    "gemma_1.1_instruct_7b_en": {
        "metadata": {
            "description": (
                "7 billion parameter, 28-layer, instruction tuned Gemma model. "
                "The 1.1 update improves model quality."
            ),
            "params": 8537680896,
            "official_name": "Gemma",
            "path": "gemma",
            "model_card": "https://www.kaggle.com/models/google/gemma",
        },
        "kaggle_handle": "kaggle://keras/gemma/keras/gemma_1.1_instruct_7b_en/3",
    },
    "code_gemma_7b_en": {
        "metadata": {
            "description": (
                "7 billion parameter, 28-layer, CodeGemma model. This model "
                "has been trained on a fill-in-the-middle (FIM) task for code "
                "completion."
            ),
            "params": 8537680896,
            "official_name": "Gemma",
            "path": "gemma",
            "model_card": "https://www.kaggle.com/models/google/gemma",
        },
        "kaggle_handle": "kaggle://keras/codegemma/keras/code_gemma_7b_en/1",
    },
    "code_gemma_instruct_7b_en": {
        "metadata": {
            "description": (
                "7 billion parameter, 28-layer, instruction tuned CodeGemma "
                "model. This model has been trained for chat use cases related "
                "to code."
            ),
            "params": 8537680896,
            "official_name": "Gemma",
            "path": "gemma",
            "model_card": "https://www.kaggle.com/models/google/gemma",
        },
        "kaggle_handle": "kaggle://keras/codegemma/keras/code_gemma_instruct_7b_en/1",
    },
    "code_gemma_1.1_instruct_7b_en": {
        "metadata": {
            "description": (
                "7 billion parameter, 28-layer, instruction tuned CodeGemma "
                "model. This model has been trained for chat use cases related "
                "to code. The 1.1 update improves model quality."
            ),
            "params": 8537680896,
            "official_name": "Gemma",
            "path": "gemma",
            "model_card": "https://www.kaggle.com/models/google/gemma",
        },
        "kaggle_handle": "kaggle://keras/codegemma/keras/code_gemma_1.1_instruct_7b_en/1",
    },
    "gemma2_2b_en": {
        "metadata": {
            "description": "2 billion parameter, 26-layer, base Gemma model.",
            "params": 2614341888,
            "official_name": "Gemma",
            "path": "gemma",
            "model_card": "https://www.kaggle.com/models/google/gemma",
        },
        "kaggle_handle": "kaggle://keras/gemma2/keras/gemma2_2b_en/1",
    },
    "gemma2_instruct_2b_en": {
        "metadata": {
            "description": "2 billion parameter, 26-layer, instruction tuned Gemma model.",
            "params": 2614341888,
            "official_name": "Gemma",
            "path": "gemma",
            "model_card": "https://www.kaggle.com/models/google/gemma",
        },
        "kaggle_handle": "kaggle://keras/gemma2/keras/gemma2_instruct_2b_en/1",
    },
    "gemma2_9b_en": {
        "metadata": {
            "description": "9 billion parameter, 42-layer, base Gemma model.",
            "params": 9241705984,
            "official_name": "Gemma",
            "path": "gemma",
            "model_card": "https://www.kaggle.com/models/google/gemma",
        },
        "kaggle_handle": "kaggle://keras/gemma2/keras/gemma2_9b_en/2",
    },
    "gemma2_instruct_9b_en": {
        "metadata": {
            "description": "9 billion parameter, 42-layer, instruction tuned Gemma model.",
            "params": 9241705984,
            "official_name": "Gemma",
            "path": "gemma",
            "model_card": "https://www.kaggle.com/models/google/gemma",
        },
        "kaggle_handle": "kaggle://keras/gemma2/keras/gemma2_instruct_9b_en/2",
    },
    "gemma2_27b_en": {
        "metadata": {
            "description": "27 billion parameter, 42-layer, base Gemma model.",
            "params": 27227128320,
            "official_name": "Gemma",
            "path": "gemma",
            "model_card": "https://www.kaggle.com/models/google/gemma",
        },
        "kaggle_handle": "kaggle://keras/gemma2/keras/gemma2_27b_en/1",
    },
    "gemma2_instruct_27b_en": {
        "metadata": {
            "description": "27 billion parameter, 42-layer, instruction tuned Gemma model.",
            "params": 27227128320,
            "official_name": "Gemma",
            "path": "gemma",
            "model_card": "https://www.kaggle.com/models/google/gemma",
        },
        "kaggle_handle": "kaggle://keras/gemma2/keras/gemma2_instruct_27b_en/1",
    },
}
