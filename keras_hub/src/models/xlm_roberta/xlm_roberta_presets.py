# Copyright 2024 The KerasHub Authors
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
"""XLM-RoBERTa model preset configurations."""

backbone_presets = {
    "xlm_roberta_base_multi": {
        "metadata": {
            "description": (
                "12-layer XLM-RoBERTa model where case is maintained. "
                "Trained on CommonCrawl in 100 languages."
            ),
            "params": 277450752,
            "official_name": "XLM-RoBERTa",
            "path": "xlm_roberta",
            "model_card": "https://github.com/facebookresearch/fairseq/blob/main/examples/xlmr/README.md",
        },
        "kaggle_handle": "kaggle://keras/xlm_roberta/keras/xlm_roberta_base_multi/2",
    },
    "xlm_roberta_large_multi": {
        "metadata": {
            "description": (
                "24-layer XLM-RoBERTa model where case is maintained. "
                "Trained on CommonCrawl in 100 languages."
            ),
            "params": 558837760,
            "official_name": "XLM-RoBERTa",
            "path": "xlm_roberta",
            "model_card": "https://github.com/facebookresearch/fairseq/blob/main/examples/xlmr/README.md",
        },
        "kaggle_handle": "kaggle://keras/xlm_roberta/keras/xlm_roberta_large_multi/2",
    },
}
