"""MobileNet preset configurations."""

backbone_presets = {
    "mobilenet_v3_small_050_imagenet": {
        "metadata": {
            "description": (
                "Small MobileNet V3 model pre-trained on the ImageNet 1k "
                "dataset at a 224x224 resolution."
            ),
            "params": 278784,
            "path": "mobilenetv3",
        },
        "kaggle_handle": "kaggle://keras/mobilenetv3/keras/mobilenet_v3_small_050_imagenet/1",
    },
}
