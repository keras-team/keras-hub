"""Swin Transformer model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "swin_tiny_patch4_window7_224": {
        "metadata": {
            "description": (
                "Swin-Tiny model pre-trained on the ImageNet 1k dataset "
                "with image resolution of 224x224."
            ),
            "params": 27_519_354,
            "path": "swin_transformer",
        },
        "kaggle_handle": "kaggle://keras/swin-transformer/keras/swin_tiny_patch4_window7_224/1",
    },
    "swin_small_patch4_window7_224": {
        "metadata": {
            "description": (
                "Swin-Small model pre-trained on the ImageNet 1k dataset "
                "with image resolution of 224x224."
            ),
            "params": 48_837_258,
            "path": "swin_transformer",
        },
        "kaggle_handle": "kaggle://keras/swin-transformer/keras/swin_small_patch4_window7_224/1",
    },
    "swin_base_patch4_window7_224": {
        "metadata": {
            "description": (
                "Swin-Base model pre-trained on the ImageNet 1k dataset "
                "with image resolution of 224x224."
            ),
            "params": 86_743_224,
            "path": "swin_transformer",
        },
        "kaggle_handle": "kaggle://keras/swin-transformer/keras/swin_base_patch4_window7_224/1",
    },
    "swin_base_patch4_window12_384": {
        "metadata": {
            "description": (
                "Swin-Base model pre-trained on the ImageNet 1k dataset "
                "with image resolution of 384x384."
            ),
            "params": 86_878_584,
            "path": "swin_transformer",
        },
        "kaggle_handle": "kaggle://keras/swin-transformer/keras/swin_base_patch4_window12_384/1",
    },
    "swin_large_patch4_window7_224": {
        "metadata": {
            "description": (
                "Swin-Large model pre-trained on the ImageNet 1k dataset "
                "with image resolution of 224x224."
            ),
            "params": 194_995_476,
            "path": "swin_transformer",
        },
        "kaggle_handle": "kaggle://keras/swin-transformer/keras/swin_large_patch4_window7_224/1",
    },
    "swin_large_patch4_window12_384": {
        "metadata": {
            "description": (
                "Swin-Large model pre-trained on the ImageNet 1k dataset "
                "with image resolution of 384x384."
            ),
            "params": 195_198_516,
            "path": "swin_transformer",
        },
        "kaggle_handle": "kaggle://keras/swin-transformer/keras/swin_large_patch4_window12_384/1",
    },
}
