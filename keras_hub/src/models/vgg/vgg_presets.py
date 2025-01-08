"""vgg preset configurations."""

backbone_presets = {
    "vgg_11_imagenet": {
        "metadata": {
            "description": (
                "11-layer vgg model pre-trained on the ImageNet 1k dataset "
                "at a 224x224 resolution."
            ),
            "params": 9220480,
            "path": "vgg",
        },
        "kaggle_handle": "kaggle://keras/vgg/keras/vgg_11_imagenet/2",
    },
    "vgg_13_imagenet": {
        "metadata": {
            "description": (
                "13-layer vgg model pre-trained on the ImageNet 1k dataset "
                "at a 224x224 resolution."
            ),
            "params": 9404992,
            "path": "vgg",
        },
        "kaggle_handle": "kaggle://keras/vgg/keras/vgg_13_imagenet/2",
    },
    "vgg_16_imagenet": {
        "metadata": {
            "description": (
                "16-layer vgg model pre-trained on the ImageNet 1k dataset "
                "at a 224x224 resolution."
            ),
            "params": 14714688,
            "path": "vgg",
        },
        "kaggle_handle": "kaggle://keras/vgg/keras/vgg_16_imagenet/2",
    },
    "vgg_19_imagenet": {
        "metadata": {
            "description": (
                "19-layer vgg model pre-trained on the ImageNet 1k dataset "
                "at a 224x224 resolution."
            ),
            "params": 20024384,
            "path": "vgg",
        },
        "kaggle_handle": "kaggle://keras/vgg/keras/vgg_19_imagenet/2",
    },
}
