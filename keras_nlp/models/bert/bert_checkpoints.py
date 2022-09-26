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

from collections import defaultdict

# TODO(jbischof): document checkpoints in keras.io and use URL in docstrings
# Metadata for loading pretrained model weights.
checkpoints = {
    "bert_tiny_uncased_en": {
        "model": "BertTiny",
        "vocabulary": "uncased_en",
        "description": (
            "Tiny size of BERT where all input is lowercased. "
            "Trained on English Wikipedia + BooksCorpus."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/bert_tiny_uncased_en/model.h5",
        "weights_hash": "c2b29fcbf8f814a0812e4ab89ef5c068",
    },
    "bert_small_uncased_en": {
        "model": "BertSmall",
        "vocabulary": "uncased_en",
        "description": (
            "Small size of BERT where all input is lowercased. "
            "Trained on English Wikipedia + BooksCorpus."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/bert_small_uncased_en/model.h5",
        "weights_hash": "08632c9479b034f342ba2c2b7afba5f7",
    },
    "bert_medium_uncased_en": {
        "model": "BertMedium",
        "vocabulary": "uncased_en",
        "description": (
            "Medium size of BERT where all input is lowercased. "
            "Trained on English Wikipedia + BooksCorpus."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/bert_medium_uncased_en/model.h5",
        "weights_hash": "bb990e1184ec6b6185450c73833cd661",
    },
    "bert_base_uncased_en": {
        "model": "BertBase",
        "vocabulary": "uncased_en",
        "description": (
            "Base size of BERT where all input is lowercased. "
            "Trained on English Wikipedia + BooksCorpus."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/bert_base_uncased_en/model.h5",
        "weights_hash": "9b2b2139f221988759ac9cdd17050b31",
    },
    "bert_base_cased_en": {
        "model": "BertBase",
        "vocabulary": "cased_en",
        "description": (
            "Base size of BERT where case is maintained. "
            "Trained on English Wikipedia + BooksCorpus."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/bert_base_cased_en/model.h5",
        "weights_hash": "f94a6cb012e18f4fb8ec92abb91864e9",
    },
    "bert_base_zh": {
        "model": "BertBase",
        "vocabulary": "zh",
        "description": ("Base size of BERT. Trained on Chinese Wikipedia."),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/bert_base_zh/model.h5",
        "weights_hash": "79afa421e386076e62ab42dad555ab0c",
    },
    "bert_base_multi_cased": {
        "model": "BertBase",
        "vocabulary": "multi_cased",
        "description": ("Base size of BERT. Trained on Chinese Wikipedia."),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/bert_base_multi_cased/model.h5",
        "weights_hash": "b0631cec0a1f2513c6cfd75ba29c33aa",
    },
    "bert_large_uncased_en": {
        "model": "BertLarge",
        "vocabulary": "uncased_en",
        "description": (
            "Large size of BERT where all input is lowercased. "
            "Trained on English Wikipedia + BooksCorpus."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/bert_large_uncased_en/model.h5",
        "weights_hash": "cc5cacc9565ef400ee4376105f40ddae",
    },
    "bert_large_cased_en": {
        "model": "BertLarge",
        "vocabulary": "cased_en",
        "description": (
            "Base size of BERT where case is maintained. "
            "Trained on English Wikipedia + BooksCorpus."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/bert_large_cased_en/model.h5",
        "weights_hash": "8b8ab82290bbf4f8db87d4f100648890",
    },
}


# Metadata for loading pretrained tokenizer vocabularies.
# We need the vocabulary_size hardcoded so we can instantiate a BERT network
# with the right embedding size without downloading the matching vocabulary.
# TODO(mattdangerw): Update our bucket structure so the vocabularies are
# stored in an independent way, rather than reading from the base model.
vocabularies = {
    "uncased_en": {
        "description": (
            "The vocabulary for BERT models trained on "
            "English Wikipedia + BooksCorpus where case is discarded."
        ),
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/bert_base_uncased_en/vocab.txt",
        "vocabulary_hash": "64800d5d8528ce344256daf115d4965e",
        "vocabulary_size": 30522,
        "lowercase": True,
    },
    "cased_en": {
        "description": (
            "The vocabulary for BERT models trained on "
            "English Wikipedia + BooksCorpus where case is maintained."
        ),
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/bert_base_cased_en/vocab.txt",
        "vocabulary_hash": "bb6ca9b42e790e5cd986bbb16444d0e0",
        "vocabulary_size": 28996,
        "lowercase": False,
    },
    "zh": {
        "description": (
            "The vocabulary for BERT models trained on Chinese Wikipedia."
        ),
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/bert_base_zh/vocab.txt",
        "vocabulary_hash": "3b5b76c4aef48ecf8cb3abaafe960f09",
        "vocabulary_size": 21128,
        "lowercase": False,
    },
    "multi_cased": {
        "description": (
            "The vocabulary for BERT models trained on trained on Wikipedias "
            "of 104 languages."
        ),
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/bert_base_multi_cased/vocab.txt",
        "vocabulary_hash": "d9d865138d17f1958502ed060ecfeeb6",
        "vocabulary_size": 119547,
        "lowercase": False,
    },
}

# Index checkpoints by arch compatibility and create lookup function
checkpoints_per_arch = defaultdict(set)
for arch, metadata in checkpoints.items():
    checkpoints_per_arch[metadata["model"]].add(arch)


def compatible_checkpoints(arch):
    """Returns a list of compatible checkpoints per arch"""
    return checkpoints_per_arch[arch]
