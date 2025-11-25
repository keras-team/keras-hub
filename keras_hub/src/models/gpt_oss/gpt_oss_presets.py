# Copyright 2024 The KerasNLP Authors
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
"""GPT-OSS preset configurations."""

backbone_presets = {
    "gpt_oss_8_7b_en": {
        "metadata": {
            "description": (
                "32-layer GPT-OSS MoE model with 7 billion "
                "active parameters and 8 experts per MoE layer."
            ),
            "params": 46702792704,
            "path": "gpt_oss",
        },
        "kaggle_handle": "kaggle://keras/gpt_oss/keras/gpt_oss_8_7b_en/1",
    },
    "gpt_oss_instruct_8_7b_en": {
        "metadata": {
            "description": (
                "Instruction fine-tuned 32-layer GPT-OSS MoE model "
                "with 7 billion active parameters and 8 experts per MoE layer."
            ),
            "params": 46702792704,
            "path": "gpt_oss",
        },
        "kaggle_handle": (
            "kaggle://keras/gpt_oss/keras/gpt_oss_instruct_8_7b_en/1"
        ),
    },
}
