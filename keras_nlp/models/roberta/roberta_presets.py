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
"""RoBERTa model preset configurations."""

backbone_presets = {
    "roberta_base_en": {
        "metadata": {
            "description": (
                "12-layer RoBERTa model where case is maintained."
                "Trained on English Wikipedia, BooksCorpus, CommonCraw, and OpenWebText."
            ),
            "params": 124052736,
            "official_name": "RoBERTa",
            "path": "roberta",
            "model_card": "https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.md",
        },
        "kaggle_handle": "kaggle://keras/roberta/keras/roberta_base_en/2",
    },
    "roberta_large_en": {
        "metadata": {
            "description": (
                "24-layer RoBERTa model where case is maintained."
                "Trained on English Wikipedia, BooksCorpus, CommonCraw, and OpenWebText."
            ),
            "params": 354307072,
            "official_name": "RoBERTa",
            "path": "roberta",
            "model_card": "https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.md",
        },
        "kaggle_handle": "kaggle://keras/roberta/keras/roberta_large_en/2",
    },
}
