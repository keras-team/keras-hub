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
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet_b0_ra_imagenet",
    },
    "efficientnet_b1_ft_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet B1 model fine-tuned on the ImageNet 1k dataset."
            ),
            "params": 7794184,
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet_b1_ft_imagenet",
    },
    "efficientnet_el_ra_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet-EdgeTPU Large model trained on the ImageNet 1k "
                "dataset with RandAugment recipe."
            ),
            "params": 10589712,
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet_b1_ft_imagenet",
    },
    "efficientnet_em_ra2_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet-EdgeTPU Medium model trained on the ImageNet 1k "
                "dataset with RandAugment2 recipe."
            ),
            "params": 6899496,
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet_b1_ft_imagenet",
    },
    "efficientnet_es_ra_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet-EdgeTPU Small model trained on the ImageNet 1k "
                "dataset with RandAugment recipe."
            ),
            "params": 5438392,
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet_b1_ft_imagenet",
    },
    "efficientnet2_rw_m_agc_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet-v2 Medium model trained on the ImageNet 1k "
                "dataset with adaptive gradient clipping."
            ),
            "params": 53236442,
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/2104.00298",
        },
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet2_rw_m_agc_imagenet",
    },
    "efficientnet2_rw_s_ra2_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet-v2 Small model trained on the ImageNet 1k "
                "dataset with RandAugment2 recipe."
            ),
            "params": 23941296,
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/2104.00298",
        },
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet2_rw_s_ra2_imagenet",
    },
    "efficientnet2_rw_t_ra2_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet-v2 Tiny model trained on the ImageNet 1k "
                "dataset with RandAugment2 recipe."
            ),
            "params": 13649388,
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/2104.00298",
        },
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet2_rw_t_ra2_imagenet",
    },
}
