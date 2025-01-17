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
"""MiT model preset configurations."""

backbone_presets_with_weights = {
    "mit_b0_ade20k_512": {
        "metadata": {
            "description": (
                "MiT (MixTransformer) model with 8 transformer blocks."
            ),
            "params": 3321962,
            "path": "mit",
        },
        "kaggle_handle": "kaggle://keras/mit/keras/mit_b0_ade20k_512/4",
    },
    "mit_b1_ade20k_512": {
        "metadata": {
            "description": (
                "MiT (MixTransformer) model with 8 transformer blocks."
            ),
            "params": 13156554,
            "path": "mit",
        },
        "kaggle_handle": "kaggle://keras/mit/keras/mit_b1_ade20k_512/4",
    },
    "mit_b2_ade20k_512": {
        "metadata": {
            "description": (
                "MiT (MixTransformer) model with 16 transformer blocks."
            ),
            "params": 24201418,
            "path": "mit",
        },
        "kaggle_handle": "kaggle://keras/mit/keras/mit_b2_ade20k_512/4",
    },
    "mit_b3_ade20k_512": {
        "metadata": {
            "description": (
                "MiT (MixTransformer) model with 28 transformer blocks."
            ),
            "params": 44077258,
            "path": "mit",
        },
        "kaggle_handle": "kaggle://keras/mit/keras/mit_b3_ade20k_512/3",
    },
    "mit_b4_ade20k_512": {
        "metadata": {
            "description": (
                "MiT (MixTransformer) model with 41 transformer blocks."
            ),
            "params": 60847818,
            "path": "mit",
        },
        "kaggle_handle": "kaggle://keras/mit/keras/mit_b4_ade20k_512/3",
    },
    "mit_b5_ade20k_640": {
        "metadata": {
            "description": (
                "MiT (MixTransformer) model with 52 transformer blocks."
            ),
            "params": 81448138,
            "path": "mit",
        },
        "kaggle_handle": "kaggle://keras/mit/keras/mit_b5_ade20k_640/3",
    },
    "mit_b0_cityscapes_1024": {
        "metadata": {
            "description": (
                "MiT (MixTransformer) model with 8 transformer blocks."
            ),
            "params": 3321962,
            "path": "mit",
        },
        "kaggle_handle": "kaggle://keras/mit/keras/mit_b0_cityscapes_1024/3",
    },
    "mit_b1_cityscapes_1024": {
        "metadata": {
            "description": (
                "MiT (MixTransformer) model with 8 transformer blocks."
            ),
            "params": 13156554,
            "path": "mit",
        },
        "kaggle_handle": "kaggle://keras/mit/keras/mit_b1_cityscapes_1024/3",
    },
    "mit_b2_cityscapes_1024": {
        "metadata": {
            "description": (
                "MiT (MixTransformer) model with 16 transformer blocks."
            ),
            "params": 24201418,
            "path": "mit",
        },
        "kaggle_handle": "kaggle://keras/mit/keras/mit_b2_cityscapes_1024/3",
    },
    "mit_b3_cityscapes_1024": {
        "metadata": {
            "description": (
                "MiT (MixTransformer) model with 28 transformer blocks."
            ),
            "params": 44077258,
            "path": "mit",
        },
        "kaggle_handle": "kaggle://keras/mit/keras/mit_b3_cityscapes_1024/3",
    },
    "mit_b4_cityscapes_1024": {
        "metadata": {
            "description": (
                "MiT (MixTransformer) model with 41 transformer blocks."
            ),
            "params": 60847818,
            "path": "mit",
        },
        "kaggle_handle": "kaggle://keras/mit/keras/mit_b4_cityscapes_1024/3",
    },
    "mit_b5_cityscapes_1024": {
        "metadata": {
            "description": (
                "MiT (MixTransformer) model with 52 transformer blocks."
            ),
            "params": 81448138,
            "path": "mit",
        },
        "kaggle_handle": "kaggle://keras/mit/keras/mit_b5_cityscapes_1024/3",
    },
}

backbone_presets = {
    **backbone_presets_with_weights,
}
