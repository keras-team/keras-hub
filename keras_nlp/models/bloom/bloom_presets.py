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
"""BLOOM model preset configurations."""

backbone_presets = {
    "bloom_560m_multi": {
        "metadata": {
            "description": (
                "24-layer Bloom model with hidden dimension of 1024. "
                "trained on 45 natural languages and 12 programming languages."
            ),
            "params": 559214592,
            "official_name": "BLOOM",
            "path": "bloom",
            "model_card": "https://huggingface.co/bigscience/bloom-560m",
        },
        "kaggle_handle": "kaggle://keras/bloom/keras/bloom_560m_multi/3",
    },
    "bloom_1.1b_multi": {
        "metadata": {
            "description": (
                "24-layer Bloom model with hidden dimension of 1536. "
                "trained on 45 natural languages and 12 programming languages."
            ),
            "params": 1065314304,
            "official_name": "BLOOM",
            "path": "bloom",
            "model_card": "https://huggingface.co/bigscience/bloom-1b1",
        },
        "kaggle_handle": "kaggle://keras/bloom/keras/bloom_1.1b_multi/1",
    },
    "bloom_1.7b_multi": {
        "metadata": {
            "description": (
                "24-layer Bloom model with hidden dimension of 2048. "
                "trained on 45 natural languages and 12 programming languages."
            ),
            "params": 1722408960,
            "official_name": "BLOOM",
            "path": "bloom",
            "model_card": "https://huggingface.co/bigscience/bloom-1b7",
        },
        "kaggle_handle": "kaggle://keras/bloom/keras/bloom_1.7b_multi/1",
    },
    "bloom_3b_multi": {
        "metadata": {
            "description": (
                "30-layer Bloom model with hidden dimension of 2560. "
                "trained on 45 natural languages and 12 programming languages."
            ),
            "params": 3002557440,
            "official_name": "BLOOM",
            "path": "bloom",
            "model_card": "https://huggingface.co/bigscience/bloom-3b",
        },
        "kaggle_handle": "kaggle://keras/bloom/keras/bloom_3b_multi/1",
    },
    "bloomz_560m_multi": {
        "metadata": {
            "description": (
                "24-layer Bloom model with hidden dimension of 1024. "
                "finetuned on crosslingual task mixture (xP3) dataset."
            ),
            "params": 559214592,
            "official_name": "BLOOMZ",
            "path": "bloom",
            "model_card": "https://huggingface.co/bigscience/bloomz-560m",
        },
        "kaggle_handle": "kaggle://keras/bloom/keras/bloomz_560m_multi/1",
    },
    "bloomz_1.1b_multi": {
        "metadata": {
            "description": (
                "24-layer Bloom model with hidden dimension of 1536. "
                "finetuned on crosslingual task mixture (xP3) dataset."
            ),
            "params": 1065314304,
            "official_name": "BLOOMZ",
            "path": "bloom",
            "model_card": "https://huggingface.co/bigscience/bloomz-1b1",
        },
        "kaggle_handle": "kaggle://keras/bloom/keras/bloomz_1.1b_multi/1",
    },
    "bloomz_1.7b_multi": {
        "metadata": {
            "description": (
                "24-layer Bloom model with hidden dimension of 2048. "
                "finetuned on crosslingual task mixture (xP3) dataset."
            ),
            "params": 1722408960,
            "official_name": "BLOOMZ",
            "path": "bloom",
            "model_card": "https://huggingface.co/bigscience/bloomz-1b7",
        },
        "kaggle_handle": "kaggle://keras/bloom/keras/bloomz_1.7b_multi/1",
    },
    "bloomz_3b_multi": {
        "metadata": {
            "description": (
                "30-layer Bloom model with hidden dimension of 2560. "
                "finetuned on crosslingual task mixture (xP3) dataset."
            ),
            "params": 3002557440,
            "official_name": "BLOOMZ",
            "path": "bloom",
            "model_card": "https://huggingface.co/bigscience/bloomz-3b",
        },
        "kaggle_handle": "kaggle://keras/bloom/keras/bloomz_3b_multi/1",
    },
}
