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
"""ALBERT model preset configurations."""


backbone_presets = {
    "albert_base_en_uncased": {
        "metadata": {
            "description": (
                "12-layer ALBERT model where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 11683584,
            "official_name": "ALBERT",
            "path": "albert",
            "model_card": "https://github.com/google-research/albert/blob/master/README.md",
        },
        "kaggle_handle": "kaggle://keras/albert/keras/albert_base_en_uncased/2",
    },
    "albert_large_en_uncased": {
        "metadata": {
            "description": (
                "24-layer ALBERT model where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 17683968,
            "official_name": "ALBERT",
            "path": "albert",
            "model_card": "https://github.com/google-research/albert/blob/master/README.md",
        },
        "kaggle_handle": "kaggle://keras/albert/keras/albert_large_en_uncased/2",
    },
    "albert_extra_large_en_uncased": {
        "metadata": {
            "description": (
                "24-layer ALBERT model where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 58724864,
            "official_name": "ALBERT",
            "path": "albert",
            "model_card": "https://github.com/google-research/albert/blob/master/README.md",
        },
        "kaggle_handle": "kaggle://keras/albert/keras/albert_extra_large_en_uncased/2",
    },
    "albert_extra_extra_large_en_uncased": {
        "metadata": {
            "description": (
                "12-layer ALBERT model where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 222595584,
            "official_name": "ALBERT",
            "path": "albert",
            "model_card": "https://github.com/google-research/albert/blob/master/README.md",
        },
        "kaggle_handle": "kaggle://keras/albert/keras/albert_extra_extra_large_en_uncased/2",
    },
}
