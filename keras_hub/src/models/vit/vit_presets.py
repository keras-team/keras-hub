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
            "path": "vit",
        },
        "kaggle_handle": "kaggle://keras/vit/keras/vit_base_patch16_224_imagenet/3",
    },
    "vit_base_patch16_384_imagenet": {
        "metadata": {
            "description": (
                "ViT-B16 model pre-trained on the ImageNet 1k dataset with "
                "image resolution of 384x384 "
            ),
            "params": 86090496,
            "path": "vit",
        },
        "kaggle_handle": "kaggle://keras/vit/keras/vit_base_patch16_384_imagenet/3",
    },
    "vit_large_patch16_224_imagenet": {
        "metadata": {
            "description": (
                "ViT-L16 model pre-trained on the ImageNet 1k dataset with "
                "image resolution of 224x224 "
            ),
            "params": 303301632,
            "path": "vit",
        },
        "kaggle_handle": "kaggle://keras/vit/keras/vit_large_patch16_224_imagenet/3",
    },
    "vit_large_patch16_384_imagenet": {
        "metadata": {
            "description": (
                "ViT-L16 model pre-trained on the ImageNet 1k dataset with "
                "image resolution of 384x384 "
            ),
            "params": 303690752,
            "path": "vit",
        },
        "kaggle_handle": "kaggle://keras/vit/keras/vit_large_patch16_384_imagenet/3",
    },
    "vit_base_patch32_384_imagenet": {
        "metadata": {
            "description": (
                "ViT-B32 model pre-trained on the ImageNet 1k dataset with "
                "image resolution of 384x384 "
            ),
            "params": 87528192,
            "path": "vit",
        },
        "kaggle_handle": "kaggle://keras/vit/keras/vit_base_patch32_384_imagenet/2",
    },
    "vit_large_patch32_384_imagenet": {
        "metadata": {
            "description": (
                "ViT-L32 model pre-trained on the ImageNet 1k dataset with "
                "image resolution of 384x384 "
            ),
            "params": 305607680,
            "path": "vit",
        },
        "kaggle_handle": "kaggle://keras/vit/keras/vit_large_patch32_384_imagenet/2",
    },
    "vit_base_patch16_224_imagenet21k": {
        "metadata": {
            "description": (
                "ViT-B16 backbone pre-trained on the ImageNet 21k dataset with "
                "image resolution of 224x224 "
            ),
            "params": 85798656,
            "path": "vit",
        },
        "kaggle_handle": "kaggle://keras/vit/keras/vit_base_patch16_224_imagenet21k/2",
    },
    "vit_base_patch32_224_imagenet21k": {
        "metadata": {
            "description": (
                "ViT-B32 backbone pre-trained on the ImageNet 21k dataset with "
                "image resolution of 224x224 "
            ),
            "params": 87455232,
            "path": "vit",
        },
        "kaggle_handle": "kaggle://keras/vit/keras/vit_base_patch32_224_imagenet21k/2",
    },
    "vit_huge_patch14_224_imagenet21k": {
        "metadata": {
            "description": (
                "ViT-H14 backbone pre-trained on the ImageNet 21k dataset with "
                "image resolution of 224x224 "
            ),
            "params": 630764800,
            "path": "vit",
        },
        "kaggle_handle": "kaggle://keras/vit/keras/vit_huge_patch14_224_imagenet21k/2",
    },
    "vit_large_patch16_224_imagenet21k": {
        "metadata": {
            "description": (
                "ViT-L16 backbone pre-trained on the ImageNet 21k dataset with "
                "image resolution of 224x224 "
            ),
            "params": 303301632,
            "path": "vit",
        },
        "kaggle_handle": "kaggle://keras/vit/keras/vit_large_patch16_224_imagenet21k/2",
    },
    "vit_large_patch32_224_imagenet21k": {
        "metadata": {
            "description": (
                "ViT-L32 backbone pre-trained on the ImageNet 21k dataset with "
                "image resolution of 224x224 "
            ),
            "params": 305510400,
            "path": "vit",
        },
        "kaggle_handle": "kaggle://keras/vit/keras/vit_large_patch32_224_imagenet21k/2",
    },
}
