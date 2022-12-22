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
"""DistilBERT model preset configurations."""

backbone_presets = {
    "distil_bert_base_en_uncased": {
        "config": {
            "vocabulary_size": 30522,
            "num_layers": 6,
            "num_heads": 12,
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "dropout": 0.1,
            "max_sequence_length": 512,
        },
        "preprocessor_config": {
            "lowercase": True,
        },
        "description": (
            "Base size of DistilBERT where all input is lowercased. "
            "Trained on English Wikipedia + BooksCorpus using BERT as the "
            "teacher model."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/distil_bert_base_en_uncased/v1/model.h5",
        "weights_hash": "6625a649572e74086d74c46b8d0b0da3",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/distil_bert_base_en_uncased/v1/vocab.txt",
        "vocabulary_hash": "64800d5d8528ce344256daf115d4965e",
    },
    "distil_bert_base_en_cased": {
        "config": {
            "vocabulary_size": 28996,
            "num_layers": 6,
            "num_heads": 12,
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "dropout": 0.1,
            "max_sequence_length": 512,
        },
        "preprocessor_config": {
            "lowercase": False,
        },
        "description": (
            "Base size of DistilBERT where case is maintained. "
            "Trained on English Wikipedia + BooksCorpus using BERT as the "
            "teacher model."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/distil_bert_base_en_cased/v1/model.h5",
        "weights_hash": "fa36aa6865978efbf85a5c8264e5eb57",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/distil_bert_base_en_cased/v1/vocab.txt",
        "vocabulary_hash": "bb6ca9b42e790e5cd986bbb16444d0e0",
    },
    "distil_bert_base_multi_cased": {
        "config": {
            "vocabulary_size": 119547,
            "num_layers": 6,
            "num_heads": 12,
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "dropout": 0.1,
            "max_sequence_length": 512,
        },
        "preprocessor_config": {
            "lowercase": False,
        },
        "description": (
            "Base size of DistilBERT. Trained on Wikipedias of 104 languages "
            "using BERT the teacher model."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/distil_bert_base_multi_cased/v1/model.h5",
        "weights_hash": "c0f11095e2a6455bd3b1a6d14800a7fa",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/distil_bert_base_multi_cased/v1/vocab.txt",
        "vocabulary_hash": "d9d865138d17f1958502ed060ecfeeb6",
    },
}
