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
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_18_imagenet/3",
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
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_50_imagenet/3",
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
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_101_imagenet/3",
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
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_152_imagenet/3",
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
        "kaggle_handle": "kaggle://kerashub/resnetv2/keras/resnet_v2_50_imagenet/3",
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
        "kaggle_handle": "kaggle://kerashub/resnetv2/keras/resnet_v2_101_imagenet/3",
    },
}
