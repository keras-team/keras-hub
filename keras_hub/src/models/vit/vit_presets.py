"""ViT model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "vit_base_patch16_224_imagenet": {
        "metadata": {
            "description": (
                "ViT-B16 model pre-trained on the ImageNet 1k dataset with "
                "image resolution of 224x224 "
            ),
            "params": 85798656,
            "official_name": "ViT",
            "path": "vit",
            "model_card": "https://www.kaggle.com/models/keras/vit",
        },
        "kaggle_handle": "kaggle://keras/vit/keras/vit_base_patch16_224_imagenet/1",
    },
    "vit_base_patch16_384_imagenet": {
        "metadata": {
            "description": (
                "ViT-B16 model pre-trained on the ImageNet 1k dataset with "
                "image resolution of 384x384 "
            ),
            "params": 86090496,
            "official_name": "ViT",
            "path": "vit",
            "model_card": "https://www.kaggle.com/models/keras/vit",
        },
        "kaggle_handle": "kaggle://keras/vit/keras/vit_base_patch16_384_imagenet/1",
    },
    "vit_large_patch16_224_imagenet": {
        "metadata": {
            "description": (
                "ViT-L16 model pre-trained on the ImageNet 1k dataset with "
                "image resolution of 224x224 "
            ),
            "params": 303301632,
            "official_name": "ViT",
            "path": "vit",
            "model_card": "https://www.kaggle.com/models/keras/vit",
        },
        "kaggle_handle": "kaggle://keras/vit/keras/vit_large_patch16_224_imagenet/1",
    },
    "vit_large_patch16_384_imagenet": {
        "metadata": {
            "description": (
                "ViT-L16 model pre-trained on the ImageNet 1k dataset with "
                "image resolution of 384x384 "
            ),
            "params": 303690752,
            "official_name": "ViT",
            "path": "vit",
            "model_card": "https://www.kaggle.com/models/keras/vit",
        },
        "kaggle_handle": "kaggle://keras/vit/keras/vit_large_patch16_384_imagenet/1",
    },
}
