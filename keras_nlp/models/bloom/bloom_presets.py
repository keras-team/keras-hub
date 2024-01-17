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
            # "description": (
            #     "2-layer BERT model where all input is lowercased. "
            #     "Trained on English Wikipedia + BooksCorpus."
            # ),
            "params": 816115712,
            "official_name": "BLOOM",
            "path": "bloom",
            # "model_card": "https://github.com/google-research/bert/blob/master/README.md",
        },
        # "kaggle_handle": "kaggle://keras/bert/keras/bert_tiny_en_uncased/2",
    },
}
