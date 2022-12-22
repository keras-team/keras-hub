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
"""DeBERTa model preset configurations."""

backbone_presets = {
    "deberta_v3_extra_small": {
        "config": {
            "vocabulary_size": 128100,
            "num_layers": 12,
            "num_heads": 6,
            "hidden_dim": 384,
            "intermediate_dim": 1536,
            "dropout": 0.1,
            "max_sequence_length": 512,
            "bucket_size": 256,
        },
        "preprocessor_config": {},
        "description": (
            "Extra small size of DeBERTaV3. "
            "Trained on English Wikipedia, BookCorpus and OpenWebText."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/deberta_v3_extra_small/v1/model.h5",
        "weights_hash": "d8e10327107e5c5e20b45548a5028619",
        "spm_proto_url": "https://storage.googleapis.com/keras-nlp/models/deberta_v3_extra_small/v1/vocab.spm",
        "spm_proto_hash": "1613fcbf3b82999c187b09c9db79b568",
    },
    "deberta_v3_small": {
        "config": {
            "vocabulary_size": 128100,
            "num_layers": 6,
            "num_heads": 12,
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "dropout": 0.1,
            "max_sequence_length": 512,
            "bucket_size": 256,
        },
        "preprocessor_config": {},
        "description": (
            "Small size of DeBERTaV3. "
            "Trained on English Wikipedia, BookCorpus and OpenWebText."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/deberta_v3_small/v1/model.h5",
        "weights_hash": "84118eb7c5a735f2061ecccaf71bb888",
        "spm_proto_url": "https://storage.googleapis.com/keras-nlp/models/deberta_v3_small/v1/vocab.spm",
        "spm_proto_hash": "1613fcbf3b82999c187b09c9db79b568",
    },
    "deberta_v3_base": {
        "config": {
            "vocabulary_size": 128100,
            "num_layers": 12,
            "num_heads": 12,
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "dropout": 0.1,
            "max_sequence_length": 512,
            "bucket_size": 256,
        },
        "preprocessor_config": {},
        "description": (
            "Base size of DeBERTaV3. "
            "Trained on English Wikipedia, BookCorpus and OpenWebText."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/deberta_v3_base/v1/model.h5",
        "weights_hash": "cebce044aeed36aec9b94e3b8a255430",
        "spm_proto_url": "https://storage.googleapis.com/keras-nlp/models/deberta_v3_base/v1/vocab.spm",
        "spm_proto_hash": "1613fcbf3b82999c187b09c9db79b568",
    },
    "deberta_v3_large": {
        "config": {
            "vocabulary_size": 128100,
            "num_layers": 24,
            "num_heads": 16,
            "hidden_dim": 1024,
            "intermediate_dim": 4096,
            "dropout": 0.1,
            "max_sequence_length": 512,
            "bucket_size": 256,
        },
        "preprocessor_config": {},
        "description": (
            "Base size of DeBERTaV3. "
            "Trained on English Wikipedia, BookCorpus and OpenWebText."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/deberta_v3_large/v1/model.h5",
        "weights_hash": "bce7690f358a9e39304f8c0ebc71a745",
        "spm_proto_url": "https://storage.googleapis.com/keras-nlp/models/deberta_v3_large/v1/vocab.spm",
        "spm_proto_hash": "1613fcbf3b82999c187b09c9db79b568",
    },
    "deberta_v3_base_multi": {
        "config": {
            "vocabulary_size": 251000,
            "num_layers": 12,
            "num_heads": 12,
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "dropout": 0.1,
            "max_sequence_length": 512,
            "bucket_size": 256,
        },
        "preprocessor_config": {},
        "description": (
            "Base size of DeBERTaV3. "
            "Trained on the 2.5TB multilingual CC100 dataset."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/deberta_v3_base_multi/v1/model.h5",
        "weights_hash": "26e5a824b26afd2ee336835bd337bbeb",
        "spm_proto_url": "https://storage.googleapis.com/keras-nlp/models/deberta_v3_base_multi/v1/vocab.spm",
        "spm_proto_hash": "b4ca07289eac48600b29529119d565e2",
    },
}
