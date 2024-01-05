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
"""DistilBERT model preset configurations."""

backbone_presets = {
    "distil_bert_base_en_uncased": {
        "metadata": {
            "description": (
                "6-layer DistilBERT model where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus using BERT as the "
                "teacher model."
            ),
            "params": 66362880,
            "official_name": "DistilBERT",
            "path": "distil_bert",
            "model_card": "https://huggingface.co/distilbert-base-uncased",
        },
        "kaggle_handle": "kaggle://keras/distil_bert/keras/distil_bert_base_en_uncased/2",
    },
    "distil_bert_base_en": {
        "metadata": {
            "description": (
                "6-layer DistilBERT model where case is maintained. "
                "Trained on English Wikipedia + BooksCorpus using BERT as the "
                "teacher model."
            ),
            "params": 65190912,
            "official_name": "DistilBERT",
            "path": "distil_bert",
            "model_card": "https://huggingface.co/distilbert-base-cased",
        },
        "kaggle_handle": "kaggle://keras/distil_bert/keras/distil_bert_base_en/2",
    },
    "distil_bert_base_multi": {
        "metadata": {
            "description": (
                "6-layer DistilBERT model where case is maintained. Trained on Wikipedias of 104 languages"
            ),
            "params": 134734080,
            "official_name": "DistilBERT",
            "path": "distil_bert",
            "model_card": "https://huggingface.co/distilbert-base-multilingual-cased",
        },
        "kaggle_handle": "kaggle://keras/distil_bert/keras/distil_bert_base_multi/2",
    },
}
