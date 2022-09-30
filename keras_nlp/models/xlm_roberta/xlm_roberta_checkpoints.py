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
    "xlm_roberta_base": {
        "model": "XLMRobertaBase",
        "vocabulary": "common_crawl",
        "description": (
            "Base size of XLM-RoBERTa with 277M parameters. Trained on "
            "the CommonCrawl dataset (100 languages)."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/xlm_roberta_base/model.h5",
        "weights_hash": "2eb6fcda5a42f0a88056213ba3d93906",
    },
    "xlm_roberta_large": {
        "model": "XLMRobertaLarge",
        "vocabulary": "common_crawl",
        "description": (
            "Large size of XLM-RoBERTa with 558M parameters. Trained on "
            "the CommonCrawl dataset (100 languages)."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/xlm_roberta_large/model.h5",
        "weights_hash": "276211827174b71751f2ce3a89da503a",
    },
}


vocabularies = {
    "common_crawl": {
        "description": (
            "The BPE SentencePiece vocabulary for XLM-RoBERTa models trained on "
            "the CommonCrawl dataset."
        ),
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/xlm_roberta_base/vocab.spm",
        "vocabulary_hash": "bf25eb5120ad92ef5c7d8596b5dc4046",
        "vocabulary_size": 250002,
    }
}

# Index checkpoints by arch compatibility.
checkpoints_per_arch = defaultdict(set)
for arch, metadata in checkpoints.items():
    checkpoints_per_arch[metadata["model"]].add(arch)


def compatible_checkpoints(arch):
    """Returns a list of compatible checkpoints per arch"""
    return checkpoints_per_arch[arch]
