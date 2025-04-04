"""Xception preset configurations."""

backbone_presets = {
    "xception_41_imagenet": {
        "metadata": {
            "description": (
                "41-layer Xception model pre-trained on ImageNet 1k."
            ),
            "params": 20861480,
            "path": "xception",
        },
        "kaggle_handle": "kaggle://keras/xception/keras/xception_41_imagenet/2",
    },
}
