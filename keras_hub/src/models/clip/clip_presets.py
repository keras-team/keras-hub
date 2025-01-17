"""CLIP model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "clip_vit_base_patch16": {
        "metadata": {
            "description": (
                "150 million parameter, 12-layer for vision and 12-layer for "
                "text, patch size of 16, CLIP model."
            ),
            "params": 149620934,
            "path": "clip",
        },
        "kaggle_handle": "kaggle://keras/clip/keras/clip_vit_base_patch16/2",
    },
    "clip_vit_base_patch32": {
        "metadata": {
            "description": (
                "151 million parameter, 12-layer for vision and 12-layer for "
                "text, patch size of 32, CLIP model."
            ),
            "params": 151277363,
            "path": "clip",
        },
        "kaggle_handle": "kaggle://keras/clip/keras/clip_vit_base_patch32/2",
    },
    "clip_vit_large_patch14": {
        "metadata": {
            "description": (
                "428 million parameter, 24-layer for vision and 12-layer for "
                "text, patch size of 14, CLIP model."
            ),
            "params": 427616770,
            "path": "clip",
        },
        "kaggle_handle": "kaggle://keras/clip/keras/clip_vit_large_patch14/2",
    },
    "clip_vit_large_patch14_336": {
        "metadata": {
            "description": (
                "428 million parameter, 24-layer for vision and 12-layer for "
                "text, patch size of 14, image size of 336, CLIP model."
            ),
            "params": 427944770,
            "path": "clip",
        },
        "kaggle_handle": "kaggle://keras/clip/keras/clip_vit_large_patch14_336/2",
    },
    "clip_vit_b_32_laion2b_s34b_b79k": {
        "metadata": {
            "description": (
                "151 million parameter, 12-layer for vision and 12-layer for "
                "text, patch size of 32, Open CLIP model."
            ),
            "params": 151277363,
            "path": "clip",
        },
        "kaggle_handle": "kaggle://keras/clip/keras/clip_vit_b_32_laion2b_s34b_b79k/2",
    },
    "clip_vit_h_14_laion2b_s32b_b79k": {
        "metadata": {
            "description": (
                "986 million parameter, 32-layer for vision and 24-layer for "
                "text, patch size of 14, Open CLIP model."
            ),
            "params": 986109698,
            "path": "clip",
        },
        "kaggle_handle": "kaggle://keras/clip/keras/clip_vit_h_14_laion2b_s32b_b79k/2",
    },
    "clip_vit_g_14_laion2b_s12b_b42k": {
        "metadata": {
            "description": (
                "1.4 billion parameter, 40-layer for vision and 24-layer for "
                "text, patch size of 14, Open CLIP model."
            ),
            "params": 1366678530,
            "path": "clip",
        },
        "kaggle_handle": "kaggle://keras/clip/keras/clip_vit_g_14_laion2b_s12b_b42k/2",
    },
    "clip_vit_bigg_14_laion2b_39b_b160k": {
        "metadata": {
            "description": (
                "2.5 billion parameter, 48-layer for vision and 32-layer for "
                "text, patch size of 14, Open CLIP model."
            ),
            "params": 2539567362,
            "path": "clip",
        },
        "kaggle_handle": "kaggle://keras/clip/keras/clip_vit_bigg_14_laion2b_39b_b160k/2",
    },
}
