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
"""XLM-RoBERTa model preset configurations."""

backbone_presets = {
    "t5_small_multi": {
        "metadata": {
            "description": (
                "8-layer T5 model. Trained on the Colossal Clean Crawled "
                "Corpus (C4)."
            ),
            "params": 0,
            "official_name": "T5",
            "path": "t5",
            "model_card": "https://github.com/google-research/text-to-text-transfer-transformer/blob/main/README.md",
        },
        "config": {
            "vocabulary_size": 32128,
            "num_layers": 6,
            "num_heads": 8,
            "hidden_dim": 512,
            "intermediate_dim": 2048,
            "key_value_dim": 64,
            "dropout": 0.1,
            "activation": "relu",
            "use_gated_activation": False,
            "layer_norm_epsilon": 1e-06,
        },
        "preprocessor_config": {},
    },
    "t5_base_multi": {
        "metadata": {
            "description": (
                "12-layer T5 model. Trained on the Colossal Clean Crawled "
                "Corpus (C4)."
            ),
            "params": 0,
            "official_name": "T5",
            "path": "t5",
            "model_card": "https://github.com/google-research/text-to-text-transfer-transformer/blob/main/README.md",
        },
        "config": {
            "vocabulary_size": 32128,
            "num_layers": 12,
            "num_heads": 12,
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "dropout": 0.1,
            "activation": "relu",
            "use_gated_activation": False,
            "layer_norm_epsilon": 1e-06,
        },
        "preprocessor_config": {},
    },
    "t5_large_multi": {
        "metadata": {
            "description": (
                "24-layer T5 model. Trained on the Colossal Clean Crawled "
                "Corpus (C4)."
            ),
            "params": 0,
            "official_name": "T5",
            "path": "t5",
            "model_card": "https://github.com/google-research/text-to-text-transfer-transformer/blob/main/README.md",
        },
        "config": {
            "vocabulary_size": 32128,
            "num_layers": 24,
            "num_heads": 16,
            "hidden_dim": 1024,
            "intermediate_dim": 4096,
            "dropout": 0.1,
            "activation": "relu",
            "use_gated_activation": False,
            "layer_norm_epsilon": 1e-06,
        },
        "preprocessor_config": {},
    },
    "flan_small_multi": {
        "metadata": {
            "description": (
                "8-layer T5 model. Trained on the Colossal Clean Crawled "
                "Corpus (C4)."
            ),
            "params": 0,
            "official_name": "T5",
            "path": "t5",
            "model_card": "https://github.com/google-research/text-to-text-transfer-transformer/blob/main/README.md",
        },
        "config": {
            "vocabulary_size": 32128,
            "num_layers": 8,
            "num_heads": 6,
            "hidden_dim": 512,
            "intermediate_dim": 1024,
            "key_value_dim": 64,
            "dropout": 0.1,
            "activation": "keras_nlp>gelu_approximate",
            "use_gated_activation": True,
            "layer_norm_epsilon": 1e-06,
        },
        "preprocessor_config": {},
    },
    "flan_base_multi": {
        "metadata": {
            "description": (
                "12-layer T5 model. Trained on the Colossal Clean Crawled "
                "Corpus (C4)."
            ),
            "params": 0,
            "official_name": "T5",
            "path": "t5",
            "model_card": "https://github.com/google-research/text-to-text-transfer-transformer/blob/main/README.md",
        },
        "config": {
            "vocabulary_size": 32128,
            "num_layers": 12,
            "num_heads": 12,
            "hidden_dim": 768,
            "intermediate_dim": 2048,
            "dropout": 0.1,
            "activation": "keras_nlp>gelu_approximate",
            "use_gated_activation": True,
            "layer_norm_epsilon": 1e-06,
        },
        "preprocessor_config": {},
    },
    "flan_large_multi": {
        "metadata": {
            "description": (
                "24-layer T5 model. Trained on the Colossal Clean Crawled "
                "Corpus (C4)."
            ),
            "params": 0,
            "official_name": "T5",
            "path": "t5",
            "model_card": "https://github.com/google-research/text-to-text-transfer-transformer/blob/main/README.md",
        },
        "config": {
            "vocabulary_size": 32128,
            "num_layers": 24,
            "num_heads": 16,
            "hidden_dim": 1024,
            "intermediate_dim": 2816,
            "dropout": 0.1,
            "activation": "keras_nlp>gelu_approximate",
            "use_gated_activation": True,
            "layer_norm_epsilon": 1e-06,
        },
        "preprocessor_config": {},
    },
}
