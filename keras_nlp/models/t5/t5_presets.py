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
"""T5 model preset configurations."""

backbone_presets = {
    "t5_small": {
        "metadata": {
            "description": (
                "'Small' configuration of T5 (60 million parameters)."
            ),
            "params": 60506624,
            "official_name": "T5-Small",
            "path": "t5",
            "model_card": "https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md",
        },
        "config": {
            "use_gated_activation": False,
            "activation": "relu",
            "hidden_dim": 2048,
            "key_value_dim": 64,
            "output_dim": 512,
            "dropout": 0.1,
            "layer_norm_epsilon": 1e-06,
            "num_heads": 8,
            "num_blocks": 6,
            "vocabulary_size": 32128,
        },
        "preprocessor_config": {},
        "weights_url": "TODO",
        "weights_hash": "TODO",
        "vocabulary_url": "TODO",
        "vocabulary_hash": "TODO",
        "merges_url": "TODO",
        "merges_hash": "TODO",
    },
    "t5_base": {
        "metadata": {
            "description": (
                "'Base' configuration of T5 (223 million parameters)."
            ),
            "params": 222903552,
            "official_name": "T5-Base",
            "path": "t5",
            "model_card": "https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md",
        },
        "config": {
            "use_gated_activation": False,
            "activation": "relu",
            "hidden_dim": 3072,
            "key_value_dim": 64,
            "output_dim": 768,
            "dropout": 0.1,
            "layer_norm_epsilon": 1e-06,
            "num_heads": 12,
            "num_blocks": 12,
            "vocabulary_size": 32128,
        },
        "preprocessor_config": {},
        "weights_url": "TODO",
        "weights_hash": "TODO",
        "vocabulary_url": "TODO",
        "vocabulary_hash": "TODO",
        "merges_url": "TODO",
        "merges_hash": "TODO",
    },
    "t5_large": {
        "metadata": {
            "description": (
                "'Large' configuration of T5 (738 million parameters)."
            ),
            "params": 737668096,
            "official_name": "T5-Large",
            "path": "t5",
            "model_card": "https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md",
        },
        "config": {
            "use_gated_activation": False,
            "activation": "relu",
            "hidden_dim": 4096,
            "key_value_dim": 64,
            "output_dim": 1024,
            "dropout": 0.1,
            "layer_norm_epsilon": 1e-06,
            "num_heads": 16,
            "num_blocks": 24,
            "vocabulary_size": 32128,
        },
        "preprocessor_config": {},
        "weights_url": "TODO",
        "weights_hash": "TODO",
        "vocabulary_url": "TODO",
        "vocabulary_hash": "TODO",
        "merges_url": "TODO",
        "merges_hash": "TODO",
    },
    "t5_3b": {
        "metadata": {
            "description": (
                "'3B' configuration of T5 (2.85 billion parameters)."
            ),
            "params": 2851598336,
            "official_name": "T5-3B",
            "path": "t5",
            "model_card": "https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md",
        },
        "config": {
            "use_gated_activation": False,
            "activation": "relu",
            "hidden_dim": 16384,
            "key_value_dim": 128,
            "output_dim": 1024,
            "dropout": 0.1,
            "layer_norm_epsilon": 1e-06,
            "num_heads": 32,
            "num_blocks": 24,
            "vocabulary_size": 32128,
        },
        "preprocessor_config": {},
        "weights_url": "TODO",
        "weights_hash": "TODO",
        "vocabulary_url": "TODO",
        "vocabulary_hash": "TODO",
        "merges_url": "TODO",
        "merges_hash": "TODO",
    },
}
