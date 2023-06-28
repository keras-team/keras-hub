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
"""Falcon model preset configurations."""

backbone_presets = {
    "falcon_base_7b": {
        "metadata": {
            "description": (
                ""
            ),
            "params": 7000000000000,
            "official_name": "Falcon",
            "path": "falcon-7b",
            "model_card": "",
        },
        "config": {
            "vocabulary_size": 0,
            "num_layers": 0,
            "num_heads": 0,
            "hidden_dim": 0,
            "intermediate_dim": 0,
            "dropout": 0.1,
            "max_sequence_length": 512,
        },
        "preprocessor_config": {
            "lowercase": True,
        },
        "weights_url": "",
        "weights_hash": "",
        "vocabulary_url": "t",
        "vocabulary_hash": "",
    },
    "falcon_base_40b": {
        "metadata": {
            "description": (
                ""
            ),
            "params": 40000000000000,
            "official_name": "Falcon-40b",
            "path": "falcon",
            "model_card": "",
        },
        "config": {
            "vocabulary_size": 0,
            "num_layers": 0,
            "num_heads": 0,
            "hidden_dim": 0,
            "intermediate_dim": 0,
            "dropout": 0.1,
            "max_sequence_length": 512,
        },
        "preprocessor_config": {
            "lowercase": False,
        },
        "weights_url": "",
        "weights_hash": "",
        "vocabulary_url": "",
        "vocabulary_hash": "",
    }
}