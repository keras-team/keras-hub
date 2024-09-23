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
"""Falcon model preset configurations."""

backbone_presets = {
    "falcon_refinedweb_1b_en": {
        "metadata": {
            "description": (
                "24-layer Falcon model (Falcon with 1B parameters), trained on "
                "350B tokens of RefinedWeb dataset."
            ),
            "params": 1311625216,
            "official_name": "Falcon",
            "path": "falcon",
            "model_card": "https://huggingface.co/tiiuae/falcon-rw-1b",
        },
        "kaggle_handle": "kaggle://keras/falcon/keras/falcon_refinedweb_1b_en/1",
    },
}
