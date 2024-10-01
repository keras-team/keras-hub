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
    "enet_b0_ra": {
        "metadata": {
            "description": (
                "EfficientNet B0 model pre-trained on the ImageNet 1k dataset "
                "with RandAugment recipe."
            ),
            "params": 11186112,  # TODO: What is this... how do I figure out what it should be?
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_18_imagenet/2",  # TODO: Where does this map to?
    },
    "enet_b1_ft": {
        "metadata": {
            "description": (
                "EfficientNet B1 model fine-trained on the ImageNet 1k dataset."
            ),
            "params": 11186112,  # TODO: What is this... how do I figure out what it should be?
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_18_imagenet/2",  # TODO: Where does this map to?
    },
    "enet_b1_pruned": {
        "metadata": {
            "description": (
                "EfficientNet B1 model pre-trained on the ImageNet 1k dataset "
                "with knapsack pruning."
            ),
            "params": 11186112,  # TODO: What is this... how do I figure out what it should be?
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_18_imagenet/2",  # TODO: Where does this map to?
    },
    "enet_b2_ra": {
        "metadata": {
            "description": (
                "EfficientNet B2 model pre-trained on the ImageNet 1k dataset "
                "with RandAugment recipe."
            ),
            "params": 11186112,  # TODO: What is this... how do I figure out what it should be?
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_18_imagenet/2",  # TODO: Where does this map to?
    },
    "enet_b2_pruned": {
        "metadata": {
            "description": (
                "EfficientNet B2 model pre-trained on the ImageNet 1k dataset "
                "with knapsack pruning."
            ),
            "params": 11186112,  # TODO: What is this... how do I figure out what it should be?
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_18_imagenet/2",  # TODO: Where does this map to?
    },
    "enet_b3_ra2": {
        "metadata": {
            "description": (
                "EfficientNet B3 model pre-trained on the ImageNet 1k dataset "
                "with RandAugment2 recipe."
            ),
            "params": 11186112,  # TODO: What is this... how do I figure out what it should be?
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_18_imagenet/2",  # TODO: Where does this map to?
    },
    "enet_b3_pruned": {
        "metadata": {
            "description": (
                "EfficientNet B3 model pre-trained on the ImageNet 1k dataset "
                "with knapsack pruning."
            ),
            "params": 11186112,  # TODO: What is this... how do I figure out what it should be?
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_18_imagenet/2",  # TODO: Where does this map to?
    },
    "enet_b4_ra2": {
        "metadata": {
            "description": (
                "EfficientNet B4 model pre-trained on the ImageNet 1k dataset "
                "with RandAugment2 recipe."
            ),
            "params": 11186112,  # TODO: What is this... how do I figure out what it should be?
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_18_imagenet/2",  # TODO: Where does this map to?
    },
    "enet_b5_sw": {
        "metadata": {
            "description": (
                "EfficientNet B5 model pre-trained on the ImageNet 12k dataset "
                " with Swin Transformer train / pretrain recipe."
            ),
            "params": 11186112,  # TODO: What is this... how do I figure out what it should be?
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_18_imagenet/2",  # TODO: Where does this map to?
    },
    "enet_b5_sw_ft": {
        "metadata": {
            "description": (
                "EfficientNet B5 model pre-trained on the ImageNet 12k dataset,"
                " fine-tuned on ImageNet 1k dataset, and with Swin Transformer "
                " train / pretrain recipe."
            ),
            "params": 11186112,  # TODO: What is this... how do I figure out what it should be?
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_18_imagenet/2",  # TODO: Where does this map to?
    },
    "enet_el_ra": {
        "metadata": {
            "description": (
                "EfficientNet-EdgeTPU large model pre-trained on the ImageNet "
                "1k dataset with RandAugment recipe."
            ),
            "params": 11186112,  # TODO: What is this... how do I figure out what it should be?
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_18_imagenet/2",  # TODO: Where does this map to?
    },
    "enet_el_pruned": {
        "metadata": {
            "description": (
                "EfficientNet-EdgeTPU large model pre-trained on the ImageNet "
                "1k dataset with knapsack pruning."
            ),
            "params": 11186112,  # TODO: What is this... how do I figure out what it should be?
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_18_imagenet/2",  # TODO: Where does this map to?
    },
    "enet_em_ra2": {
        "metadata": {
            "description": (
                "EfficientNet-EdgeTPU medium model pre-trained on the ImageNet "
                "1k dataset with RandAugment2 recipe."
            ),
            "params": 11186112,  # TODO: What is this... how do I figure out what it should be?
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_18_imagenet/2",  # TODO: Where does this map to?
    },
    "enet_es_ra": {
        "metadata": {
            "description": (
                "EfficientNet-EdgeTPU small model pre-trained on the ImageNet "
                "1k dataset with RandAugment recipe."
            ),
            "params": 11186112,  # TODO: What is this... how do I figure out what it should be?
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_18_imagenet/2",  # TODO: Where does this map to?
    },
    "enet_es_pruned": {
        "metadata": {
            "description": (
                "EfficientNet-EdgeTPU small model pre-trained on the ImageNet "
                "1k dataset with knapsack pruning."
            ),
            "params": 11186112,  # TODO: What is this... how do I figure out what it should be?
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_18_imagenet/2",  # TODO: Where does this map to?
    },
    "enet_b0_ra4_e3600_r224": {
        "metadata": {
            "description": (
                "EfficientNet b0 model pre-trained on the ImageNet 1k dataset "
                "using hyper-parameters inspired by MobileNet-V4 small."
            ),
            "params": 11186112,  # TODO: What is this... how do I figure out what it should be?
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_18_imagenet/2",  # TODO: Where does this map to?
    },
    "enet_b1_ra4_e3600_r240": {
        "metadata": {
            "description": (
                "EfficientNet b1 model pre-trained on the ImageNet 1k dataset "
                "using hyper-parameters inspired by MobileNet-V4 small."
            ),
            "params": 11186112,  # TODO: What is this... how do I figure out what it should be?
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_18_imagenet/2",  # TODO: Where does this map to?
    },
    "enet2_rw_m_agc": {
        "metadata": {
            "description": (
                "EfficientNet-v2 medium model pre-trained on the ImageNet 1k "
                "dataset using adaptive gradient clipping."
            ),
            "params": 11186112,  # TODO: What is this... how do I figure out what it should be?
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/2104.00298",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_18_imagenet/2",  # TODO: Where does this map to?
    },
    "enet2_rw_s_ra2": {
        "metadata": {
            "description": (
                "EfficientNet-v2 small model pre-trained on the ImageNet 1k "
                "dataset using RandAugment2 recipe."
            ),
            "params": 11186112,  # TODO: What is this... how do I figure out what it should be?
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/2104.00298",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_18_imagenet/2",  # TODO: Where does this map to?
    },
    "enet2_rw_t_ra2": {
        "metadata": {
            "description": (
                "EfficientNet-v2 tiny model pre-trained on the ImageNet 1k "
                "dataset using RandAugment2 recipe."
            ),
            "params": 11186112,  # TODO: What is this... how do I figure out what it should be?
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/2104.00298",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_18_imagenet/2",  # TODO: Where does this map to?
    },
}
