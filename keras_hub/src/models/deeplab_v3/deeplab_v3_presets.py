"""DeepLabV3 preset configurations."""

backbone_presets = {
    "deeplabv3_plus_resnet50_pascalvoc": {
        "metadata": {
            "description": (
                "DeepLabV3+ model with ResNet50 as image encoder and trained on "
                "augmented Pascal VOC dataset by Semantic Boundaries Dataset(SBD)"
                "which is having categorical accuracy of 90.01 and 0.63 Mean IoU."
            ),
            "params": 39190656,
            "official_name": "DeepLabV3",
            "path": "deeplabv3",
            "model_card": "https://arxiv.org/abs/1802.02611",
        },
        "kaggle_handle": "kaggle://keras/deeplabv3/keras/deeplabv3_plus_resnet50_pascalvoc/3",
    },
}
