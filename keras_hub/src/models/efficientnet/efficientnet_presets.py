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
                "EfficientNet B1 model fine-trained on the ImageNet 1k dataset."
            ),
            "params": 7794184,
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet_b1_ft_imagenet",
    },
    "efficientnet_lite0_ra_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet-Lite model fine-trained on the ImageNet 1k dataset "
                "with RandAugment recipe."
            ),
            "params": 4652008,
            "official_name": "EfficientNet",
            "path": "efficientnet",
            "model_card": "https://arxiv.org/abs/1905.11946",
        },
        "kaggle_handle": "kaggle://keras/efficientnet/keras/efficientnet_lite0_ra_imagenet",
    },
}
