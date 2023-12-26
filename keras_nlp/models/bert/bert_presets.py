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
"""BERT model preset configurations."""

# TODO(jbischof): document presets in keras.io and use URL in docstrings
# Metadata for loading pretrained model weights.
backbone_presets = {
    "bert_tiny_en_uncased": {
        "metadata": {
            "description": (
                "2-layer BERT model where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 4385920,
            "official_name": "BERT",
            "path": "bert",
            "model_card": "https://github.com/google-research/bert/blob/master/README.md",
        },
        "kaggle_handle": "gs://keras-nlp-kaggle/bert_tiny_en_uncased",
    },
    "bert_small_en_uncased": {
        "metadata": {
            "description": (
                "4-layer BERT model where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 28763648,
            "official_name": "BERT",
            "path": "bert",
            "model_card": "https://github.com/google-research/bert/blob/master/README.md",
        },
        "kaggle_handle": "gs://keras-nlp-kaggle/bert_small_en_uncased",
    },
    "bert_medium_en_uncased": {
        "metadata": {
            "description": (
                "8-layer BERT model where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 41373184,
            "official_name": "BERT",
            "path": "bert",
            "model_card": "https://github.com/google-research/bert/blob/master/README.md",
        },
        "kaggle_handle": "gs://keras-nlp-kaggle/bert_medium_en_uncased",
    },
    "bert_base_en_uncased": {
        "metadata": {
            "description": (
                "12-layer BERT model where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 109482240,
            "official_name": "BERT",
            "path": "bert",
            "model_card": "https://github.com/google-research/bert/blob/master/README.md",
        },
        "kaggle_handle": "gs://keras-nlp-kaggle/bert_base_en_uncased",
    },
    "bert_base_en": {
        "metadata": {
            "description": (
                "12-layer BERT model where case is maintained. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 108310272,
            "official_name": "BERT",
            "path": "bert",
            "model_card": "https://github.com/google-research/bert/blob/master/README.md",
        },
        "kaggle_handle": "gs://keras-nlp-kaggle/bert_base_en",
    },
    "bert_base_zh": {
        "metadata": {
            "description": (
                "12-layer BERT model. Trained on Chinese Wikipedia."
            ),
            "params": 102267648,
            "official_name": "BERT",
            "path": "bert",
            "model_card": "https://github.com/google-research/bert/blob/master/README.md",
        },
        "kaggle_handle": "gs://keras-nlp-kaggle/bert_base_zh",
    },
    "bert_base_multi": {
        "metadata": {
            "description": (
                "12-layer BERT model where case is maintained. Trained on trained on Wikipedias of 104 languages"
            ),
            "params": 177853440,
            "official_name": "BERT",
            "path": "bert",
            "model_card": "https://github.com/google-research/bert/blob/master/README.md",
        },
        "kaggle_handle": "gs://keras-nlp-kaggle/bert_base_multi",
    },
    "bert_large_en_uncased": {
        "metadata": {
            "description": (
                "24-layer BERT model where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 335141888,
            "official_name": "BERT",
            "path": "bert",
            "model_card": "https://github.com/google-research/bert/blob/master/README.md",
        },
        "kaggle_handle": "gs://keras-nlp-kaggle/bert_large_en_uncased",
    },
    "bert_large_en": {
        "metadata": {
            "description": (
                "24-layer BERT model where case is maintained. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 333579264,
            "official_name": "BERT",
            "path": "bert",
            "model_card": "https://github.com/google-research/bert/blob/master/README.md",
        },
        "kaggle_handle": "gs://keras-nlp-kaggle/bert_large_en",
    },
}

classifier_presets = {
    "bert_tiny_en_uncased_sst2": {
        "metadata": {
            "description": (
                "The bert_tiny_en_uncased backbone model fine-tuned on the SST-2 sentiment analysis dataset."
            ),
            "params": 4385920,
            "official_name": "BERT",
            "path": "bert",
            "model_card": "https://github.com/google-research/bert/blob/master/README.md",
        },
        "kaggle_handle": "gs://keras-nlp-kaggle/bert_tiny_en_uncased_sst2",
    }
}
