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
"""EfficientNet preset configurations."""

backbone_presets = {
    "efficientnet_b0_ra_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet B0 model pre-trained on the ImageNet 1k dataset "
                "with RandAugment recipe."
            ),
            "params": 5288548,
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://keras/efficientnetv1/keras/efficientnet_b0_ra_imagenet",
    },
    "efficientnet_b1_ft_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet B1 model fine-trained on the ImageNet 1k dataset."
            ),
            "params": 7794184,
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://keras/efficientnetv1/keras/efficientnet_b1_ft_imagenet",
    },
}
