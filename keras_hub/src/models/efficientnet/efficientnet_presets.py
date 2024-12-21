"""EfficientNet preset configurations."""

backbone_presets = {
    "efficientnet_b0_ra_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet B0 model pre-trained on the ImageNet 1k dataset "
                "with RandAugment recipe."
            ),
            "params": 5288548,
            "path": "efficientnet",
        },
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet_b0_ra_imagenet/2",
    },
    "efficientnet_b0_ra4_e3600_r224_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet B0 model pre-trained on the ImageNet 1k dataset "
                "by Ross Wightman. Trained with timm scripts using "
                "hyper-parameters inspired by the MobileNet-V4 small, mixed "
                "with go-to hparams from timm and 'ResNet Strikes Back'."
            ),
            "params": 5288548,
            "path": "efficientnet",
        },
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet_b0_ra4_e3600_r224_imagenet/2",
    },
    "efficientnet_b1_ft_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet B1 model fine-tuned on the ImageNet 1k dataset."
            ),
            "params": 7794184,
            "path": "efficientnet",
        },
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet_b1_ft_imagenet/5",
    },
    "efficientnet_b1_ra4_e3600_r240_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet B1 model pre-trained on the ImageNet 1k dataset "
                "by Ross Wightman. Trained with timm scripts using "
                "hyper-parameters inspired by the MobileNet-V4 small, mixed "
                "with go-to hparams from timm and 'ResNet Strikes Back'."
            ),
            "params": 7794184,
            "path": "efficientnet",
        },
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet_b1_ra4_e3600_r240_imagenet/2",
    },
    "efficientnet_b2_ra_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet B2 model pre-trained on the ImageNet 1k dataset "
                "with RandAugment recipe."
            ),
            "params": 9109994,
            "path": "efficientnet",
        },
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet_b2_ra_imagenet/2",
    },
    "efficientnet_b3_ra2_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet B3 model pre-trained on the ImageNet 1k dataset "
                "with RandAugment2 recipe."
            ),
            "params": 12233232,
            "path": "efficientnet",
        },
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet_b3_ra2_imagenet/2",
    },
    "efficientnet_b4_ra2_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet B4 model pre-trained on the ImageNet 1k dataset "
                "with RandAugment2 recipe."
            ),
            "params": 19341616,
            "path": "efficientnet",
        },
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet_b4_ra2_imagenet/2",
    },
    "efficientnet_b5_sw_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet B5 model pre-trained on the ImageNet 12k dataset "
                "by Ross Wightman. Based on Swin Transformer train / pretrain "
                "recipe with modifications (related to both DeiT and ConvNeXt "
                "recipes)."
            ),
            "params": 30389784,
            "path": "efficientnet",
        },
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet_b5_sw_imagenet/2",
    },
    "efficientnet_b5_sw_ft_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet B5 model pre-trained on the ImageNet 12k dataset "
                "and fine-tuned on ImageNet-1k by Ross Wightman. Based on Swin "
                "Transformer train / pretrain recipe with modifications "
                "(related to both DeiT and ConvNeXt recipes)."
            ),
            "params": 30389784,
            "path": "efficientnet",
        },
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet_b5_sw_ft_imagenet/2",
    },
    "efficientnet_el_ra_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet-EdgeTPU Large model trained on the ImageNet 1k "
                "dataset with RandAugment recipe."
            ),
            "params": 10589712,
            "path": "efficientnet",
        },
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet_b1_ft_imagenet/5",
    },
    "efficientnet_em_ra2_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet-EdgeTPU Medium model trained on the ImageNet 1k "
                "dataset with RandAugment2 recipe."
            ),
            "params": 6899496,
            "path": "efficientnet",
        },
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet_b1_ft_imagenet/5",
    },
    "efficientnet_es_ra_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet-EdgeTPU Small model trained on the ImageNet 1k "
                "dataset with RandAugment recipe."
            ),
            "params": 5438392,
            "path": "efficientnet",
        },
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet_b1_ft_imagenet/5",
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
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet2_rw_m_agc_imagenet/2",
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
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet2_rw_s_ra2_imagenet/2",
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
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet2_rw_t_ra2_imagenet/2",
    },
    "efficientnet_lite0_ra_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet-Lite model fine-trained on the ImageNet 1k "
                "dataset with RandAugment recipe."
            ),
            "params": 4652008,
            "path": "efficientnet",
        },
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet_lite0_ra_imagenet/2",
    },
}
