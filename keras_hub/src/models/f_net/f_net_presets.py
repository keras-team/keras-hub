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
"""FNet model preset configurations."""

backbone_presets = {
    "f_net_base_en": {
        "metadata": {
            "description": (
                "12-layer FNet model where case is maintained. "
                "Trained on the C4 dataset."
            ),
            "params": 82861056,
            "official_name": "FNet",
            "path": "f_net",
            "model_card": "https://github.com/google-research/google-research/blob/master/f_net/README.md",
        },
        "kaggle_handle": "kaggle://keras/f_net/keras/f_net_base_en/2",
    },
    "f_net_large_en": {
        "metadata": {
            "description": (
                "24-layer FNet model where case is maintained. "
                "Trained on the C4 dataset."
            ),
            "params": 236945408,
            "official_name": "FNet",
            "path": "f_net",
            "model_card": "https://github.com/google-research/google-research/blob/master/f_net/README.md",
        },
        "kaggle_handle": "kaggle://keras/f_net/keras/f_net_large_en/2",
    },
}
