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
"""ResNet preset configurations."""

backbone_presets = {
    "resnet_18_imagenet": {
        "metadata": {
            "description": (
                "18-layer ResNet model pre-trained on the ImageNet 1k dataset "
                "at a 224x224 resolution."
            ),
            "params": 11186112,
            "official_name": "ResNet",
            "path": "resnet",
            "model_card": "https://arxiv.org/abs/2110.00476",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_18_imagenet/2",
    },
    "resnet_50_imagenet": {
        "metadata": {
            "description": (
                "50-layer ResNet model pre-trained on the ImageNet 1k dataset "
                "at a 224x224 resolution."
            ),
            "params": 23561152,
            "official_name": "ResNet",
            "path": "resnet",
            "model_card": "https://arxiv.org/abs/2110.00476",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_50_imagenet/2",
    },
    "resnet_101_imagenet": {
        "metadata": {
            "description": (
                "101-layer ResNet model pre-trained on the ImageNet 1k dataset "
                "at a 224x224 resolution."
            ),
            "params": 42605504,
            "official_name": "ResNet",
            "path": "resnet",
            "model_card": "https://arxiv.org/abs/2110.00476",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_101_imagenet/2",
    },
    "resnet_152_imagenet": {
        "metadata": {
            "description": (
                "152-layer ResNet model pre-trained on the ImageNet 1k dataset "
                "at a 224x224 resolution."
            ),
            "params": 58295232,
            "official_name": "ResNet",
            "path": "resnet",
            "model_card": "https://arxiv.org/abs/2110.00476",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_152_imagenet/2",
    },
    "resnet_v2_50_imagenet": {
        "metadata": {
            "description": (
                "50-layer ResNetV2 model pre-trained on the ImageNet 1k "
                "dataset at a 224x224 resolution."
            ),
            "params": 23561152,
            "official_name": "ResNet",
            "path": "resnet",
            "model_card": "https://arxiv.org/abs/2110.00476",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv2/keras/resnet_v2_50_imagenet/2",
    },
    "resnet_v2_101_imagenet": {
        "metadata": {
            "description": (
                "101-layer ResNetV2 model pre-trained on the ImageNet 1k "
                "dataset at a 224x224 resolution."
            ),
            "params": 42605504,
            "official_name": "ResNet",
            "path": "resnet",
            "model_card": "https://arxiv.org/abs/2110.00476",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv2/keras/resnet_v2_101_imagenet/2",
    },
}
