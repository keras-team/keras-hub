"""Swin Transformer model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "swin_tiny_patch4_window7_224": {
        "metadata": {
            "description": (
                "Swin Transformer with tiny architecture "
                "(C=96, depths=[2, 2, 6, 2]), 4x4 patch size, 7x7 window "
                "size, pre-trained on ImageNet-1K at 224x224 resolution. "
                "Developed by Microsoft Research."
            ),
            "params": 27519354,
            "official_name": "SwinTransformer",
            "path": "swin_transformer",
        },
        "kaggle_handle": "kaggle://keras/swin_transformer/keras/swin_tiny_patch4_window7_224/1",
    },
    "swin_small_patch4_window7_224": {
        "metadata": {
            "description": (
                "Swin Transformer with small architecture "
                "(C=96, depths=[2, 2, 18, 2]), 4x4 patch size, 7x7 window "
                "size, pre-trained on ImageNet-1K at 224x224 resolution. "
                "Developed by Microsoft Research."
            ),
            "params": 48837258,
            "official_name": "SwinTransformer",
            "path": "swin_transformer",
        },
        "kaggle_handle": "kaggle://keras/swin_transformer/keras/swin_small_patch4_window7_224/1",
    },
    "swin_base_patch4_window7_224": {
        "metadata": {
            "description": (
                "Swin Transformer with base architecture "
                "(C=128, depths=[2, 2, 18, 2]), 4x4 patch size, 7x7 window "
                "size, pre-trained on ImageNet-1K at 224x224 resolution. "
                "Developed by Microsoft Research."
            ),
            "params": 86743224,
            "official_name": "SwinTransformer",
            "path": "swin_transformer",
        },
        "kaggle_handle": "kaggle://keras/swin_transformer/keras/swin_base_patch4_window7_224/1",
    },
    "swin_base_patch4_window12_384": {
        "metadata": {
            "description": (
                "Swin Transformer with base architecture "
                "(C=128, depths=[2, 2, 18, 2]), 4x4 patch size, 12x12 "
                "window size, pre-trained on ImageNet-1K at 384x384 "
                "resolution. Developed by Microsoft Research."
            ),
            "params": 86878584,
            "official_name": "SwinTransformer",
            "path": "swin_transformer",
        },
        "kaggle_handle": "kaggle://keras/swin_transformer/keras/swin_base_patch4_window12_384/1",
    },
    "swin_large_patch4_window7_224": {
        "metadata": {
            "description": (
                "Swin Transformer with large architecture "
                "(C=192, depths=[2, 2, 18, 2]), 4x4 patch size, 7x7 window "
                "size, pre-trained on ImageNet-1K at 224x224 resolution. "
                "Developed by Microsoft Research."
            ),
            "params": 194995476,
            "official_name": "SwinTransformer",
            "path": "swin_transformer",
        },
        "kaggle_handle": "kaggle://keras/swin_transformer/keras/swin_large_patch4_window7_224/1",
    },
    "swin_large_patch4_window12_384": {
        "metadata": {
            "description": (
                "Swin Transformer with large architecture "
                "(C=192, depths=[2, 2, 18, 2]), 4x4 patch size, 12x12 "
                "window size, pre-trained on ImageNet-1K at 384x384 "
                "resolution. Developed by Microsoft Research."
            ),
            "params": 195198516,
            "official_name": "SwinTransformer",
            "path": "swin_transformer",
        },
        "kaggle_handle": "kaggle://keras/swin_transformer/keras/swin_large_patch4_window12_384/1",
    },
}
