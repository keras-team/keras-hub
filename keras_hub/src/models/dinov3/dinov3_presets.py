"""DINOV3 model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "dinov3_vit_small_lvd1689m": {
        "metadata": {
            "description": (
                "Vision Transformer (small-sized model) trained on LVD-1689M "
                "using DINOv3."
            ),
            "params": 21_600_000,
            "path": "dinov3",
        },
        "kaggle_handle": "kaggle://keras/dinov3/keras/dinov3_vit_small_lvd1689m/1",
    },
    "dinov3_vit_small_plus_lvd1689m": {
        "metadata": {
            "description": (
                "Vision Transformer (small-plus-sized model) trained on "
                "LVD-1689M using DINOv3."
            ),
            "params": 29_000_000,
            "path": "dinov3",
        },
        "kaggle_handle": "kaggle://keras/dinov3/keras/dinov3_vit_small_plus_lvd1689m/1",
    },
    "dinov3_vit_base_lvd1689m": {
        "metadata": {
            "description": (
                "Vision Transformer (base-sized model) trained on LVD-1689M "
                "using DINOv3."
            ),
            "params": 86_000_000,
            "path": "dinov3",
        },
        "kaggle_handle": "kaggle://keras/dinov3/keras/dinov3_vit_base_lvd1689m/1",
    },
    "dinov3_vit_large_lvd1689m": {
        "metadata": {
            "description": (
                "Vision Transformer (large-sized model) trained on LVD-1689M "
                "using DINOv3."
            ),
            "params": 300_000_000,
            "path": "dinov3",
        },
        "kaggle_handle": "kaggle://keras/dinov3/keras/dinov3_vit_large_lvd1689m/1",
    },
    "dinov3_vit_huge_plus_lvd1689m": {
        "metadata": {
            "description": (
                "Vision Transformer (huge-plus-sized model) trained on "
                "LVD-1689M using DINOv3."
            ),
            "params": 840_000_000,
            "path": "dinov3",
        },
        "kaggle_handle": "kaggle://keras/dinov3/keras/dinov3_vit_huge_plus_lvd1689m/1",
    },
    "dinov3_vit_7b_lvd1689m": {
        "metadata": {
            "description": (
                "Vision Transformer (7B-sized model) trained on LVD-1689M "
                "using DINOv3."
            ),
            "params": 6_700_000_000,
            "path": "dinov3",
        },
        "kaggle_handle": "kaggle://keras/dinov3/keras/dinov3_vit_7b_lvd1689m/1",
    },
    "dinov3_vit_large_sat493m": {
        "metadata": {
            "description": (
                "Vision Transformer (large-sized model) trained on SAT-493M "
                "using DINOv3."
            ),
            "params": 300_000_000,
            "path": "dinov3",
        },
        "kaggle_handle": "kaggle://keras/dinov3/keras/dinov3_vit_large_sat493m/1",
    },
    "dinov3_vit_7b_sat493m": {
        "metadata": {
            "description": (
                "Vision Transformer (7B-sized model) trained on SAT-493M "
                "using DINOv3."
            ),
            "params": 6_700_000_000,
            "path": "dinov3",
        },
        "kaggle_handle": "kaggle://keras/dinov3/keras/dinov3_vit_7b_sat493m/1",
    },
}
