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
"""ELECTRA model preset configurations."""

backbone_presets = {
    "electra_small_discriminator_uncased_en": {
        "metadata": {
            "description": (
                "12-layer small ELECTRA discriminator model. All inputs are "
                "lowercased. Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 13548800,
            "official_name": "ELECTRA",
            "path": "electra",
            "model_card": "https://github.com/google-research/electra",
        },
        "kaggle_handle": "kaggle://keras/electra/keras/electra_small_discriminator_uncased_en/1",
    },
    "electra_small_generator_uncased_en": {
        "metadata": {
            "description": (
                "12-layer small ELECTRA generator model. All inputs are "
                "lowercased. Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 13548800,
            "official_name": "ELECTRA",
            "path": "electra",
            "model_card": "https://github.com/google-research/electra",
        },
        "kaggle_handle": "kaggle://keras/electra/keras/electra_small_generator_uncased_en/1",
    },
    "electra_base_discriminator_uncased_en": {
        "metadata": {
            "description": (
                "12-layer base ELECTRA discriminator model. All inputs are "
                "lowercased. Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 109482240,
            "official_name": "ELECTRA",
            "path": "electra",
            "model_card": "https://github.com/google-research/electra",
        },
        "kaggle_handle": "kaggle://keras/electra/keras/electra_base_discriminator_uncased_en/1",
    },
    "electra_base_generator_uncased_en": {
        "metadata": {
            "description": (
                "12-layer base ELECTRA generator model. All inputs are "
                "lowercased. Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 33576960,
            "official_name": "ELECTRA",
            "path": "electra",
            "model_card": "https://github.com/google-research/electra",
        },
        "kaggle_handle": "kaggle://keras/electra/keras/electra_base_generator_uncased_en/1",
    },
    "electra_large_discriminator_uncased_en": {
        "metadata": {
            "description": (
                "24-layer large ELECTRA discriminator model. All inputs are "
                "lowercased. Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 335141888,
            "official_name": "ELECTRA",
            "path": "electra",
            "model_card": "https://github.com/google-research/electra",
        },
        "kaggle_handle": "kaggle://keras/electra/keras/electra_large_discriminator_uncased_en/1",
    },
    "electra_large_generator_uncased_en": {
        "metadata": {
            "description": (
                "24-layer large ELECTRA generator model. All inputs are "
                "lowercased. Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 51065344,
            "official_name": "ELECTRA",
            "path": "electra",
            "model_card": "https://github.com/google-research/electra",
        },
        "kaggle_handle": "kaggle://keras/electra/keras/electra_large_generator_uncased_en/1",
    },
}
