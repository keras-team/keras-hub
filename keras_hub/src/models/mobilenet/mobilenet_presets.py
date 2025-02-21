"""MobileNet preset configurations."""

backbone_presets = {
    "mobilenetv3_small_050": {
        "metadata": {
            "description": (
                "Small Mobilenet V3 model pre-trained on the ImageNet 1k "
                "dataset at a 224x224 resolution. Has half channel multiplier."
            ),
            "params": 278784,
            "path": "mobilenetv3",
        },
        "kaggle_handle": "kaggle://keras/mobilenetv3/keras/mobilenet_v3_small_050_imagenet/1",
    },
    "mobilenetv3_small_100": {
        "metadata": {
            "description": (
                "Small Mobilenet V3 model pre-trained on the ImageNet 1k "
                "dataset at a 224x224 resolution. Has baseline channel "
                "multiplier."
            ),
            "params": 939120,
            "path": "mobilenetv3",
        },
        "kaggle_handle": "kaggle://keras/mobilenetv3/keras/mobilenet_v3_small_100_imagenet/1",
    },
    "mobilenetv3_large_100.ra_in1k": {
        "metadata": {
            "description": (
                "Large Mobilenet V3 model pre-trained on the ImageNet 1k "
                "dataset at a 224x224 resolution. Has baseline channel "
                "multiplier."
            ),
            "params": 2996352,
            "path": "mobilenetv3",
        },
        "kaggle_handle": "kaggle://keras/mobilenetv3/keras/mobilenet_v3_large_100_imagenet/1",
    },
    "mobilenetv3_large_100.miil_in21k_ft_in1k": {
        "metadata": {
            "description": (
                "Large Mobilenet V3 model pre-trained on the ImageNet 21k "
                "dataset at a 224x224 resolution. Has baseline channel "
                "multiplier."
            ),
            "params": 2996352,
            "path": "mobilenetv3",
        },
        "kaggle_handle": "kaggle://keras/mobilenetv3/keras/mobilenet_v3_large_100_imagenet_21k/1",
    },
}
