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
