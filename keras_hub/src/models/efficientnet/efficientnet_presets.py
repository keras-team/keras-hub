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
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet_b0_ra_imagenet/1",
    },
    "efficientnet_b0_ra4_e3600_r224_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet B0 model pre-trained on the ImageNet 1k dataset by"
                " Ross Wightman. Trained with timm scripts using hyper-parameters"
                " inspired by the MobileNet-V4 small, mixed with go-to hparams "
                'from timm and "ResNet Strikes Back".'
            ),
            "params": 5288548,
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet_b0_ra4_e3600_r224_imagenet/1",
    },
    "efficientnet_b1_ft_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet B1 model pre-trained on the ImageNet 1k dataset."
            ),
            "params": 7794184,
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet_b1_ft_imagenet/1",
    },
    "efficientnet_b1_ra4_e3600_r240_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet B1 model pre-trained on the ImageNet 1k dataset by"
                " Ross Wightman. Trained with timm scripts using hyper-parameters"
                " inspired by the MobileNet-V4 small, mixed with go-to hparams "
                'from timm and "ResNet Strikes Back".'
            ),
            "params": 7794184,
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet_b1_ra4_e3600_r240_imagenet/1",
    },
    "efficientnet_b2_ra_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet B2 model pre-trained on the ImageNet 1k dataset "
                "with RandAugment recipe."
            ),
            "params": 9109994,
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet_b2_ra_imagenet/1",
    },
    "efficientnet_b3_ra2_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet B3 model pre-trained on the ImageNet 1k dataset "
                "with RandAugment2 recipe."
            ),
            "params": 12233232,
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet_b3_ra2_imagenet/1",
    },
    "efficientnet_b4_ra2_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet B4 model pre-trained on the ImageNet 1k dataset "
                "with RandAugment2 recipe."
            ),
            "params": 19341616,
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet_b4_ra2_imagenet/1",
    },
    "efficientnet_b5_sw_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet B5 model pre-trained on the ImageNet 12k dataset "
                "by Ross Wightman. Based on Swin Transformer train / pretrain "
                "recipe with modifications (related to both DeiT and ConvNeXt recipes)."
            ),
            "params": 30389784,
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet_b5_sw_imagenet/1",
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
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet_b5_sw_ft_imagenet/1",
    },
}
