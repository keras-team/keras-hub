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

checkpoints = {
    "distilbert_base_uncased_en": {
        "model": "DistilBertBase",
        "vocabulary": "uncased_en",
        "description": (
            "Base size of DistilBERT where all input is lowercased. "
            "Trained on English Wikipedia + BooksCorpus."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/distilbert_base_uncased_en/model.h5",
        "weights_hash": "6625a649572e74086d74c46b8d0b0da3",
    },
    "distilbert_base_cased_en": {
        "model": "DistilBertBase",
        "vocabulary": "cased_en",
        "description": (
            "Base size of DistilBERT where case is maintained. "
            "Trained on English Wikipedia + BooksCorpus."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/distilbert_base_cased_en/model.h5",
        "weights_hash": "fa36aa6865978efbf85a5c8264e5eb57",
    },
    "distilbert_base_multi_cased": {
        "model": "DistilBertBase",
        "vocabulary": "multi_cased",
        "description": (
            "Base size of DistilBERT. Trained on Wikipedias of 104 languages."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/distilbert_base_multi_cased/model.h5",
        "weights_hash": "c0f11095e2a6455bd3b1a6d14800a7fa",
    },
}


# Metadata for loading pretrained tokenizer vocabularies.
# TODO: Vocabularies of DistilBERT are the same as BERT's, including the
# hash. We can have a common folder for both BERT and DistilBERT vocabularies.
vocabularies = {
    "uncased_en": {
        "description": (
            "The vocabulary for DistilBERT models trained on "
            "English Wikipedia + BooksCorpus where case is discarded."
        ),
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/distilbert_base_uncased_en/vocab.txt",
        "vocabulary_hash": "64800d5d8528ce344256daf115d4965e",
        "vocabulary_size": 30522,
        "lowercase": True,
    },
    "cased_en": {
        "description": (
            "The vocabulary for DistilBERT models trained on "
            "English Wikipedia + BooksCorpus where case is maintained."
        ),
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/distilbert_base_cased_en/vocab.txt",
        "vocabulary_hash": "bb6ca9b42e790e5cd986bbb16444d0e0",
        "vocabulary_size": 28996,
        "lowercase": False,
    },
    "multi_cased": {
        "description": (
            "The vocabulary for DistilBERT models trained on Wikipedias "
            "of 104 languages."
        ),
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/distilbert_base_multi_cased/vocab.txt",
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
