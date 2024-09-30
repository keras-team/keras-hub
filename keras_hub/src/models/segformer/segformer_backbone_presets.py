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
"""SegFormerBackbone model preset configurations."""

presets_no_weights = {
    "segformer_b0_backbone": {
        "metadata": {
            "description": ("SegFormerBackbone model with MiTB0 encoder."),
            "params": 3719027,
            "official_name": "SegFormerB0Backbone",
            "path": "segformer_b0_backbone",
        },
        "kaggle_handle": "kaggle://TBA",
    },
    "segformer_b1_backbone": {
        "metadata": {
            "description": ("SegFormerBackbone model with MiTB1 encoder."),
            "params": 13682643,
            "official_name": "SegFormerB1Backbone",
            "path": "segformer_b1_backbone",
        },
        "kaggle_handle": "kaggle://TBA",
    },
    "segformer_b2_backbone": {
        "metadata": {
            "description": ("SegFormerBackbone model with MiTB2 encoder."),
            "params": 24727507,
            "official_name": "SegFormerB2Backbone",
            "path": "segformer_b2_backbone",
        },
        "kaggle_handle": "kaggle://TBA",
    },
    "segformer_b3_backbone": {
        "metadata": {
            "description": ("SegFormerBackbone model with MiTB3 encoder."),
            "params": 44603347,
            "official_name": "SegFormerB3Backbone",
            "path": "segformer_b3_backbone",
        },
        "kaggle_handle": "kaggle://TBA",
    },
    "segformer_b4_backbone": {
        "metadata": {
            "description": ("SegFormerBackbone model with MiTB4 encoder."),
            "params": 61373907,
            "official_name": "SegFormerB4Backbone",
            "path": "segformer_b4_backbone",
        },
        "kaggle_handle": "kaggle://TBA",
    },
    "segformer_b5_backbone": {
        "metadata": {
            "description": ("SegFormerBackbone model with MiTB5 encoder."),
            "params": 81974227,
            "official_name": "SegFormerB5Backbone",
            "path": "segformer_b5_backbone",
        },
        "kaggle_handle": "kaggle://TBA",
    },
}


presets = {**presets_no_weights}
