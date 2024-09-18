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
"""BART model preset configurations."""

backbone_presets = {
    "bart_base_en": {
        "metadata": {
            "description": (
                "6-layer BART model where case is maintained. "
                "Trained on BookCorpus, English Wikipedia and CommonCrawl."
            ),
            "params": 139417344,
            "official_name": "BART",
            "path": "bart",
            "model_card": "https://github.com/facebookresearch/fairseq/blob/main/examples/bart/README.md",
        },
        "kaggle_handle": "kaggle://keras/bart/keras/bart_base_en/2",
    },
    "bart_large_en": {
        "metadata": {
            "description": (
                "12-layer BART model where case is maintained. "
                "Trained on BookCorpus, English Wikipedia and CommonCrawl."
            ),
            "params": 406287360,
            "official_name": "BART",
            "path": "bart",
            "model_card": "https://github.com/facebookresearch/fairseq/blob/main/examples/bart/README.md",
        },
        "config": {
            "vocabulary_size": 50265,
            "num_layers": 12,
            "num_heads": 16,
            "hidden_dim": 1024,
            "intermediate_dim": 4096,
            "dropout": 0.1,
            "max_sequence_length": 1024,
        },
        "kaggle_handle": "kaggle://keras/bart/keras/bart_large_en/2",
    },
    "bart_large_en_cnn": {
        "metadata": {
            "description": (
                "The `bart_large_en` backbone model fine-tuned on the CNN+DM "
                "summarization dataset."
            ),
            "params": 406287360,
            "official_name": "BART",
            "path": "bart",
            "model_card": "https://github.com/facebookresearch/fairseq/blob/main/examples/bart/README.md",
        },
        "config": {
            "vocabulary_size": 50264,
            "num_layers": 12,
            "num_heads": 16,
            "hidden_dim": 1024,
            "intermediate_dim": 4096,
            "dropout": 0.1,
            "max_sequence_length": 1024,
        },
        "kaggle_handle": "kaggle://keras/bart/keras/bart_large_en_cnn/2",
    },
}
