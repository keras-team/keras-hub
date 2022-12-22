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
"""BERT model preset configurations."""

# TODO(jbischof): document presets in keras.io and use URL in docstrings
# Metadata for loading pretrained model weights.
backbone_presets = {
    "bert_tiny_en_uncased": {
        "config": {
            "vocabulary_size": 30522,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 128,
            "intermediate_dim": 512,
            "dropout": 0.1,
            "max_sequence_length": 512,
            "num_segments": 2,
        },
        "preprocessor_config": {
            "lowercase": True,
        },
        "description": (
            "Tiny size of BERT where all input is lowercased. "
            "Trained on English Wikipedia + BooksCorpus."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/bert_tiny_en_uncased/v1/model.h5",
        "weights_hash": "c2b29fcbf8f814a0812e4ab89ef5c068",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/bert_tiny_en_uncased/v1/vocab.txt",
        "vocabulary_hash": "64800d5d8528ce344256daf115d4965e",
    },
    "bert_small_en_uncased": {
        "config": {
            "vocabulary_size": 30522,
            "num_layers": 4,
            "num_heads": 8,
            "hidden_dim": 512,
            "intermediate_dim": 2048,
            "dropout": 0.1,
            "max_sequence_length": 512,
            "num_segments": 2,
        },
        "preprocessor_config": {
            "lowercase": True,
        },
        "description": (
            "Small size of BERT where all input is lowercased. "
            "Trained on English Wikipedia + BooksCorpus."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/bert_small_en_uncased/v1/model.h5",
        "weights_hash": "08632c9479b034f342ba2c2b7afba5f7",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/bert_small_en_uncased/v1/vocab.txt",
        "vocabulary_hash": "64800d5d8528ce344256daf115d4965e",
    },
    "bert_medium_en_uncased": {
        "config": {
            "vocabulary_size": 30522,
            "num_layers": 8,
            "num_heads": 8,
            "hidden_dim": 512,
            "intermediate_dim": 2048,
            "dropout": 0.1,
            "max_sequence_length": 512,
            "num_segments": 2,
        },
        "preprocessor_config": {
            "lowercase": True,
        },
        "description": (
            "Medium size of BERT where all input is lowercased. "
            "Trained on English Wikipedia + BooksCorpus."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/bert_medium_en_uncased/v1/model.h5",
        "weights_hash": "bb990e1184ec6b6185450c73833cd661",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/bert_medium_en_uncased/v1/vocab.txt",
        "vocabulary_hash": "64800d5d8528ce344256daf115d4965e",
    },
    "bert_base_en_uncased": {
        "config": {
            "vocabulary_size": 30522,
            "num_layers": 12,
            "num_heads": 12,
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "dropout": 0.1,
            "max_sequence_length": 512,
            "num_segments": 2,
        },
        "preprocessor_config": {
            "lowercase": True,
        },
        "description": (
            "Base size of BERT where all input is lowercased. "
            "Trained on English Wikipedia + BooksCorpus."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/bert_base_en_uncased/v1/model.h5",
        "weights_hash": "9b2b2139f221988759ac9cdd17050b31",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/bert_base_en_uncased/v1/vocab.txt",
        "vocabulary_hash": "64800d5d8528ce344256daf115d4965e",
    },
    "bert_base_en_cased": {
        "config": {
            "vocabulary_size": 28996,
            "num_layers": 12,
            "num_heads": 12,
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "dropout": 0.1,
            "max_sequence_length": 512,
            "num_segments": 2,
        },
        "preprocessor_config": {
            "lowercase": False,
        },
        "description": (
            "Base size of BERT where case is maintained. "
            "Trained on English Wikipedia + BooksCorpus."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/bert_base_en_cased/v1/model.h5",
        "weights_hash": "f94a6cb012e18f4fb8ec92abb91864e9",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/bert_base_en_cased/v1/vocab.txt",
        "vocabulary_hash": "bb6ca9b42e790e5cd986bbb16444d0e0",
    },
    "bert_base_zh": {
        "config": {
            "vocabulary_size": 21128,
            "num_layers": 12,
            "num_heads": 12,
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "dropout": 0.1,
            "max_sequence_length": 512,
            "num_segments": 2,
        },
        "preprocessor_config": {
            "lowercase": False,
        },
        "description": ("Base size of BERT. Trained on Chinese Wikipedia."),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/bert_base_zh/v1/model.h5",
        "weights_hash": "79afa421e386076e62ab42dad555ab0c",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/bert_base_zh/v1/vocab.txt",
        "vocabulary_hash": "3b5b76c4aef48ecf8cb3abaafe960f09",
    },
    "bert_base_multi_cased": {
        "config": {
            "vocabulary_size": 119547,
            "num_layers": 12,
            "num_heads": 12,
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "dropout": 0.1,
            "max_sequence_length": 512,
            "num_segments": 2,
        },
        "preprocessor_config": {
            "lowercase": False,
        },
        "description": (
            "Base size of BERT. Trained on trained on Wikipedias of 104 "
            "languages."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/bert_base_multi_cased/v1/model.h5",
        "weights_hash": "b0631cec0a1f2513c6cfd75ba29c33aa",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/bert_base_multi_cased/v1/vocab.txt",
        "vocabulary_hash": "d9d865138d17f1958502ed060ecfeeb6",
    },
    "bert_large_en_uncased": {
        "config": {
            "vocabulary_size": 30522,
            "num_layers": 24,
            "num_heads": 16,
            "hidden_dim": 1024,
            "intermediate_dim": 4096,
            "dropout": 0.1,
            "max_sequence_length": 512,
            "num_segments": 2,
        },
        "preprocessor_config": {
            "lowercase": True,
        },
        "description": (
            "Large size of BERT where all input is lowercased. "
            "Trained on English Wikipedia + BooksCorpus."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/bert_large_en_uncased/v1/model.h5",
        "weights_hash": "cc5cacc9565ef400ee4376105f40ddae",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/bert_large_en_uncased/v1/vocab.txt",
        "vocabulary_hash": "64800d5d8528ce344256daf115d4965e",
    },
    "bert_large_en_cased": {
        "config": {
            "vocabulary_size": 28996,
            "num_layers": 24,
            "num_heads": 16,
            "hidden_dim": 1024,
            "intermediate_dim": 4096,
            "dropout": 0.1,
            "max_sequence_length": 512,
            "num_segments": 2,
        },
        "preprocessor_config": {
            "lowercase": False,
        },
        "description": (
            "Base size of BERT where case is maintained. "
            "Trained on English Wikipedia + BooksCorpus."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/bert_large_en_cased/v1/model.h5",
        "weights_hash": "8b8ab82290bbf4f8db87d4f100648890",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/bert_large_en_cased/v1/vocab.txt",
        "vocabulary_hash": "bb6ca9b42e790e5cd986bbb16444d0e0",
    },
}

classifier_presets = {
    "bert_tiny_en_uncased_sst2": {
        "config": {
            "backbone": {
                "class_name": "keras_nlp>BertBackbone",
                "config": {
                    "vocabulary_size": 30522,
                    "hidden_dim": 128,
                    "intermediate_dim": 512,
                    "num_layers": 2,
                    "num_heads": 2,
                    "max_sequence_length": 512,
                    "num_segments": 2,
                    "dropout": 0.1,
                },
            },
            "num_classes": 2,
            "dropout": 0.1,
        },
        "preprocessor_config": {
            "lowercase": True,
        },
        "description": (
            "bert_tiny_en_uncased backbone fine-tuned on the glue/sst2 dataset."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/bert_tiny_en_uncased_sst2/v1/model.h5",
        "weights_hash": "1f9c2d59f9e229e08f3fbd44239cfb0b",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/bert_tiny_en_uncased_sst2/v1/vocab.txt",
        "vocabulary_hash": "64800d5d8528ce344256daf115d4965e",
    }
}
