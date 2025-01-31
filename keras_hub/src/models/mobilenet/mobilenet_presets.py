"""MobileNet preset configurations."""

backbone_presets = {
    "mobilenetv3_small_050": {
        "metadata": {
            "description": (
                "Small MObilenet V3 model pre-trained on the ImageNet 1k "
                "dataset at a 224x224 resolution."
            ),
            "official_name": "MobileNet",
            "path": "mobilenet3",
        },
        "kaggle_handle": "kaggle://keras/mobilenet/keras/mobilenetv3_small_050",
    },
}
