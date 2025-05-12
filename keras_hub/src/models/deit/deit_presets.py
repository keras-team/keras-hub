"""DeiT model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "deit-base-distilled-patch16-384_imagenet": {
        "metadata": {
            "description": (
                "DeiT-B16 model pre-trained on the ImageNet 1k dataset with "
                "image resolution of 384x384 "
            ),
            "params": 86092032,
            "path": "deit",
        },
        "kaggle_handle": "kaggle://keras/deit/keras/deit_base_distilled_patch16_384_imagenet/1",
    },
    "deit-base-distilled-patch16-224_imagenet": {
        "metadata": {
            "description": (
                "DeiT-B16 model pre-trained on the ImageNet 1k dataset with "
                "image resolution of 224x224 "
            ),
            "params": 85800192,
            "path": "deit",
        },
        "kaggle_handle": "kaggle://keras/deit/keras/deit_base_distilled_patch16_224_imagenet/1",
    },
    "deit-tiny-distilled-patch16-224_imagenet": {
        "metadata": {
            "description": (
                "DeiT-T16 model pre-trained on the ImageNet 1k dataset with "
                "image resolution of 224x224 "
            ),
            "params": 5524800,
            "path": "deit",
        },
        "kaggle_handle": "kaggle://keras/deit/keras/deit_tiny_distilled_patch16_224_imagenet/1",
    },
    "deit-small-distilled-patch16-224_imagenet": {
        "metadata": {
            "description": (
                "DeiT-S16 model pre-trained on the ImageNet 1k dataset with "
                "image resolution of 224x224 "
            ),
            "params": 21666432,
            "path": "deit",
        },
        "kaggle_handle": "kaggle://keras/deit/keras/deit_small_distilled_patch16_224_imagenet/1",
    },
}
