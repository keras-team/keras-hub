"""DeepLabV3 preset configurations."""

backbone_presets = {
    "deeplab_v3_plus_resnet50_pascalvoc": {
        "metadata": {
            "description": (
                "DeepLabV3+ model with ResNet50 as image encoder and trained "
                "on augmented Pascal VOC dataset by Semantic Boundaries "
                "Dataset(SBD) which is having categorical accuracy of 90.01 "
                "and 0.63 Mean IoU."
            ),
            "params": 39190656,
            "path": "deeplab_v3",
        },
        "kaggle_handle": "kaggle://keras/deeplabv3plus/keras/deeplab_v3_plus_resnet50_pascalvoc/4",
    },
}
