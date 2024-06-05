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
"""Phi-3 model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "phi3_mini_4k_instruct_en": {
        "metadata": {
            "description": (
                "3.8 billion parameters, 32 layers, 4k context length, Phi-3 "
                "model. The model was trained using the Phi-3 datasets. This "
                "dataset includes both synthetic data and filtered publicly "
                "available website data, with an emphasis on high-quality and "
                "reasoning-dense properties."
            ),
            "params": 3821079552,
            "official_name": "Phi-3",
            "path": "phi3",
            "model_card": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct",
        },
        "kaggle_handle": "kaggle://keras/phi3/keras/phi3_mini_4k_instruct_en",
    },
    "phi3_mini_128k_instruct_en": {
        "metadata": {
            "description": (
                "3.8 billion parameters, 32 layers, 128k context length, Phi-3 "
                "model. The model was trained using the Phi-3 datasets. This "
                "dataset includes both synthetic data and filtered publicly "
                "available website data, with an emphasis on high-quality and "
                "reasoning-dense properties."
            ),
            "params": 3821079552,
            "official_name": "Phi-3",
            "path": "phi3",
            "model_card": "https://huggingface.co/microsoft/Phi-3-mini-128k-instruct",
        },
        "kaggle_handle": "kaggle://keras/phi3/keras/phi3_mini_128k_instruct_en",
    },
}
