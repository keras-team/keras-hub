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
            "dropout_rate": 0.1,
            "layer_norm_epsilon": 1e-06,
            "num_heads": 8,
            "num_blocks": 6,
            "vocab_size": 32128,
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
