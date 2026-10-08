"""VideoPrism model preset configurations."""

backbone_presets = {
    "videoprism_public_v1_base": {
        "metadata": {
            "description": (
                "114 million parameter, 12-layer ViT-B, 16-frame, 288x288 "
                "resolution, video-only encoder for "
                "spatiotemporal representation."
            ),
            "params": 114000000,
            "path": "video_prism",
        },
        "kaggle_handle": "kaggle://keras/videoprism/keras/videoprism_public_v1_base/1",
    },
    "videoprism_public_v1_large": {
        "metadata": {
            "description": (
                "354 million parameter, 24-layer ViT-L, 16-frame, 288x288 "
                "resolution, video-only encoder for "
                "spatiotemporal representation."
            ),
            "params": 354000000,
            "path": "video_prism",
        },
        "kaggle_handle": "kaggle://keras/videoprism/keras/videoprism_public_v1_large/1",
    },
    "videoprism_lvt_public_v1_base": {
        "metadata": {
            "description": (
                "248 million parameter, 12-layer ViT-B video encoder + text "
                "encoder, 16-frame, 288x288 resolution, for multimodal "
                "video-language tasks."
            ),
            "params": 248000000,
            "path": "video_prism",
        },
        "kaggle_handle": "kaggle://keras/videoprism/keras/videoprism_lvt_public_v1_base/1",
    },
    "videoprism_lvt_public_v1_large": {
        "metadata": {
            "description": (
                "580 million parameter, 24-layer ViT-L video encoder + text "
                "encoder, 16-frame, 288x288 resolution, for multimodal "
                "video-language tasks."
            ),
            "params": 580000000,
            "path": "video_prism",
        },
        "kaggle_handle": "kaggle://keras/videoprism/keras/videoprism_lvt_public_v1_large/1",
    },
}
