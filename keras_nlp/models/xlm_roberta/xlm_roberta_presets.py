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
"""XLM-RoBERTa model preset configurations."""

backbone_presets = {
    "xlm_roberta_base": {
        "config": {
            "vocabulary_size": 250002,
            "num_layers": 12,
            "num_heads": 12,
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "dropout": 0.1,
            "max_sequence_length": 512,
        },
        "preprocessor_config": {},
        "description": (
            "Base size of XLM-RoBERTa. "
            "Trained on the CommonCrawl dataset (100 languages)."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/xlm_roberta_base/v1/model.h5",
        "weights_hash": "2eb6fcda5a42f0a88056213ba3d93906",
        "spm_proto_url": "https://storage.googleapis.com/keras-nlp/models/xlm_roberta_base/v1/vocab.spm",
        "spm_proto_hash": "bf25eb5120ad92ef5c7d8596b5dc4046",
    },
    "xlm_roberta_large": {
        "config": {
            "vocabulary_size": 250002,
            "num_layers": 24,
            "num_heads": 16,
            "hidden_dim": 1024,
            "intermediate_dim": 4096,
            "dropout": 0.1,
            "max_sequence_length": 512,
        },
        "preprocessor_config": {},
        "description": (
            "Large size of XLM-RoBERTa. "
            "Trained on the CommonCrawl dataset (100 languages)."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/xlm_roberta_large/v1/model.h5",
        "weights_hash": "276211827174b71751f2ce3a89da503a",
        "spm_proto_url": "https://storage.googleapis.com/keras-nlp/models/xlm_roberta_large/v1/vocab.spm",
        "spm_proto_hash": "bf25eb5120ad92ef5c7d8596b5dc4046",
    },
}
