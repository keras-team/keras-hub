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
"""OPT model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "opt_125m_en": {
        "metadata": {
            "description": (
                "12-layer OPT model where case in maintained. Trained on "
                "BookCorpus, CommonCrawl, Pile, and PushShift.io corpora."
            ),
            "params": 125237760,
            "official_name": "OPT",
            "path": "opt",
            "model_card": "https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/model_card.md",
        },
        "config": {
            "vocabulary_size": 50272,
            "num_layers": 12,
            "num_heads": 12,
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "dropout": 0.1,
            "max_sequence_length": 2048,
        },
        "preprocessor_config": {},
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/opt_125m_en/v1/model.h5",
        "weights_hash": "63e444998982e48da4a1a3970f4c6203",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/opt_125m_en/v1/vocab.json",
        "vocabulary_hash": "cf410ee085c5c69c957bb1f6d8456596",
        "merges_url": "https://storage.googleapis.com/keras-nlp/models/opt_125m_en/v1/merges.txt",
        "merges_hash": "75a37753dd7a28a2c5df80c28bf06e4e",
    },
    # We skip the 350m checkpoint because it does not match the structure of
    # other checkpoints.
    "opt_1.3b_en": {
        "metadata": {
            "description": (
                "24-layer OPT model where case in maintained. Trained on "
                "BookCorpus, CommonCrawl, Pile, and PushShift.io corpora."
            ),
            "params": 1315753984,
            "official_name": "OPT",
            "path": "opt",
            "model_card": "https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/model_card.md",
        },
        "config": {
            "vocabulary_size": 50272,
            "num_layers": 24,
            "num_heads": 32,
            "hidden_dim": 2048,
            "intermediate_dim": 8192,
            "dropout": 0.1,
            "max_sequence_length": 2048,
        },
        "preprocessor_config": {},
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/opt_1.3b_en/v1/model.h5",
        "weights_hash": "0365ac8483e99a912c9770521909ecce",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/opt_1.3b_en/v1/vocab.json",
        "vocabulary_hash": "cf410ee085c5c69c957bb1f6d8456596",
        "merges_url": "https://storage.googleapis.com/keras-nlp/models/opt_1.3b_en/v1/merges.txt",
        "merges_hash": "75a37753dd7a28a2c5df80c28bf06e4e",
    },
    "opt_2.7b_en": {
        "metadata": {
            "description": (
                "32-layer OPT model where case in maintained. Trained on "
                "BookCorpus, CommonCrawl, Pile, and PushShift.io corpora."
            ),
            "params": 2700000000,
            "official_name": "OPT",
            "path": "opt",
            "model_card": "https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/model_card.md",
        },
        "config": {
            "vocabulary_size": 50272,
            "num_layers": 32,
            "num_heads": 32,
            "hidden_dim": 2560,
            "intermediate_dim": 10240,
            "dropout": 0.1,
            "max_sequence_length": 2048,
        },
        "preprocessor_config": {},
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/opt_2.7b_en/v1/model.h5",
        "weights_hash": "af56da9206a95b9287356955c5bc14e7",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/opt_2.7b_en/v1/vocab.json",
        "vocabulary_hash": "cf410ee085c5c69c957bb1f6d8456596",
        "merges_url": "https://storage.googleapis.com/keras-nlp/models/opt_2.7b_en/v1/merges.txt",
        "merges_hash": "75a37753dd7a28a2c5df80c28bf06e4e",
    },
    "opt_6.7b_en": {
        "metadata": {
            "description": (
                "32-layer OPT model where case in maintained. Trained on "
                "BookCorpus, CommonCrawl, Pile, and PushShift.io corpora."
            ),
            "params": 6700000000,
            "official_name": "OPT",
            "path": "opt",
            "model_card": "https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/model_card.md",
        },
        "config": {
            "vocabulary_size": 50272,
            "num_layers": 32,
            "num_heads": 32,
            "hidden_dim": 4096,
            "intermediate_dim": 16384,
            "dropout": 0.1,
            "max_sequence_length": 2048,
        },
        "preprocessor_config": {},
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/opt_6.7b_en/v1/model.h5",
        "weights_hash": "543120fbe601b70e6ec04cc909781e21",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/opt_6.7b_en/v1/vocab.json",
        "vocabulary_hash": "cf410ee085c5c69c957bb1f6d8456596",
        "merges_url": "https://storage.googleapis.com/keras-nlp/models/opt_6.7b_en/v1/merges.txt",
        "merges_hash": "75a37753dd7a28a2c5df80c28bf06e4e",
    },
}
