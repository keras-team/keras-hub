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
"""DenseNet preset configurations."""

backbone_presets = {
    "densenet_121_imagenet": {
        "metadata": {
            "description": (
                "121-layer DenseNet model pre-trained on the ImageNet 1k dataset "
                "at a 224x224 resolution."
            ),
            "params": 7037504,
            "official_name": "DenseNet",
            "path": "densenet",
            "model_card": "https://arxiv.org/abs/1608.06993",
        },
        "kaggle_handle": "kaggle://kerashub/densenet/keras/densenet_121_imagenet",
    },
    "densenet_169_imagenet": {
        "metadata": {
            "description": (
                "169-layer DenseNet model pre-trained on the ImageNet 1k dataset "
                "at a 224x224 resolution."
            ),
            "params": 12642880,
            "official_name": "DenseNet",
            "path": "densenet",
            "model_card": "https://arxiv.org/abs/1608.06993",
        },
        "kaggle_handle": "kaggle://kerashub/densenet/keras/densenet_169_imagenet",
    },
    "densenet_201_imagenet": {
        "metadata": {
            "description": (
                "201-layer DenseNet model pre-trained on the ImageNet 1k dataset "
                "at a 224x224 resolution."
            ),
            "params": 18321984,
            "official_name": "DenseNet",
            "path": "densenet",
            "model_card": "https://arxiv.org/abs/1608.06993",
        },
        "kaggle_handle": "kaggle://kerashub/densenet/keras/densenet_201_imagenet",
    },
}
