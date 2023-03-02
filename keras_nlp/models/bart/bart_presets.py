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
        "config": {
            "vocabulary_size": 50265,
            "num_layers": 6,
            "num_heads": 12,
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "dropout": 0.1,
            "max_sequence_length": 1024,
        },
        "preprocessor_config": {},
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/bart_base_en/v1/model.h5",
        "weights_hash": "5b59403f0cafafbd89680e0785791163",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/bart_base_en/v1/vocab.json",
        "vocabulary_hash": "be4d3c6f3f5495426b2c03b334334354",
        "merges_url": "https://storage.googleapis.com/keras-nlp/models/bart_base_en/v1/merges.txt",
        "merges_hash": "75a37753dd7a28a2c5df80c28bf06e4e",
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
        "preprocessor_config": {},
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/bart_large_en/v1/model.h5",
        "weights_hash": "6bfe7e591af8c5699ce6f9f18753af9a",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/bart_large_en/v1/vocab.json",
        "vocabulary_hash": "cf410ee085c5c69c957bb1f6d8456596",
        "merges_url": "https://storage.googleapis.com/keras-nlp/models/bart_large_en/v1/merges.txt",
        "merges_hash": "75a37753dd7a28a2c5df80c28bf06e4e",
    },
}
