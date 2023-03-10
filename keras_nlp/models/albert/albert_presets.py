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
        "config": {
            "vocabulary_size": 30000,
            "num_layers": 12,
            "num_heads": 12,
            "num_groups": 1,
            "num_inner_repetitions": 1,
            "embedding_dim": 128,
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "dropout": 0.0,
            "max_sequence_length": 512,
            "num_segments": 2,
        },
        "preprocessor_config": {},
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/albert_base_en_uncased/v1/model.h5",
        "weights_hash": "b83ccf3418dd84adc569324183176813",
        "spm_proto_url": "https://storage.googleapis.com/keras-nlp/models/albert_base_en_uncased/v1/vocab.spm",
        "spm_proto_hash": "73e62ff8e90f951f24c8b907913039a5",
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
        "config": {
            "vocabulary_size": 30000,
            "num_layers": 24,
            "num_heads": 16,
            "num_groups": 1,
            "num_inner_repetitions": 1,
            "embedding_dim": 128,
            "hidden_dim": 1024,
            "intermediate_dim": 4096,
            "dropout": 0,
            "max_sequence_length": 512,
            "num_segments": 2,
        },
        "preprocessor_config": {},
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/albert_large_en_uncased/v1/model.h5",
        "weights_hash": "c7754804efb245f06dd6e7ced32e082c",
        "spm_proto_url": "https://storage.googleapis.com/keras-nlp/models/albert_large_en_uncased/v1/vocab.spm",
        "spm_proto_hash": "73e62ff8e90f951f24c8b907913039a5",
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
        "config": {
            "vocabulary_size": 30000,
            "num_layers": 24,
            "num_heads": 16,
            "num_groups": 1,
            "num_inner_repetitions": 1,
            "embedding_dim": 128,
            "hidden_dim": 2048,
            "intermediate_dim": 8192,
            "dropout": 0,
            "max_sequence_length": 512,
            "num_segments": 2,
        },
        "preprocessor_config": {},
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/albert_extra_large_en_uncased/v1/model.h5",
        "weights_hash": "713209be8aadfa614fd79f18c9aeb16d",
        "spm_proto_url": "https://storage.googleapis.com/keras-nlp/models/albert_extra_large_en_uncased/v1/vocab.spm",
        "spm_proto_hash": "73e62ff8e90f951f24c8b907913039a5",
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
        "config": {
            "vocabulary_size": 30000,
            "num_layers": 12,
            "num_heads": 64,
            "num_groups": 1,
            "num_inner_repetitions": 1,
            "embedding_dim": 128,
            "hidden_dim": 4096,
            "intermediate_dim": 16384,
            "dropout": 0,
            "max_sequence_length": 512,
            "num_segments": 2,
        },
        "preprocessor_config": {},
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/albert_extra_extra_large_en_uncased/v1/model.h5",
        "weights_hash": "a835177b692fb6a82139f94c66db2f22",
        "spm_proto_url": "https://storage.googleapis.com/keras-nlp/models/albert_extra_extra_large_en_uncased/v1/vocab.spm",
        "spm_proto_hash": "73e62ff8e90f951f24c8b907913039a5",
    },
}
