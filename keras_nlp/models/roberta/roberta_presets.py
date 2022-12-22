# Copyright 2022 The KerasNLP Authors
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
    "roberta_base": {
        "config": {
            "vocabulary_size": 50265,
            "num_layers": 12,
            "num_heads": 12,
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "dropout": 0.1,
            "max_sequence_length": 512,
        },
        "preprocessor_config": {},
        "description": (
            "Base size of RoBERTa where case is maintained."
            "Trained on a 160 GB English dataset."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/roberta_base/v1/model.h5",
        "weights_hash": "958eede1c7edaa9308e027be18fde7a8",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/roberta_base/v1/vocab.json",
        "vocabulary_hash": "be4d3c6f3f5495426b2c03b334334354",
        "merges_url": "https://storage.googleapis.com/keras-nlp/models/roberta_base/v1/merges.txt",
        "merges_hash": "75a37753dd7a28a2c5df80c28bf06e4e",
    },
    "roberta_large": {
        "config": {
            "vocabulary_size": 50265,
            "num_layers": 24,
            "num_heads": 16,
            "hidden_dim": 1024,
            "intermediate_dim": 4096,
            "dropout": 0.1,
            "max_sequence_length": 512,
        },
        "preprocessor_config": {},
        "description": (
            "Large size of RoBERTa where case is maintained."
            "Trained on a 160 GB English dataset."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/roberta_large/v1/model.h5",
        "weights_hash": "1978b864c317a697fe62a894d3664f14",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/roberta_large/v1/vocab.json",
        "vocabulary_hash": "be4d3c6f3f5495426b2c03b334334354",
        "merges_url": "https://storage.googleapis.com/keras-nlp/models/roberta_large/v1/merges.txt",
        "merges_hash": "75a37753dd7a28a2c5df80c28bf06e4e",
    },
}
