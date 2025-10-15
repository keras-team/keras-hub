"""DepthAnything model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "depth_anything_v2_small": {
        "metadata": {
            "description": (
                "Small variant of Depth Anything V2 monocular depth estimation "
                "(MDE) model trained on synthetic labeled images and real "
                "unlabeled images."
            ),
            "params": 25_311_169,
            "path": "depth_anything",
        },
        "kaggle_handle": "kaggle://keras/depth-anything/keras/depth_anything_v2_small/1",
    },
    "depth_anything_v2_base": {
        "metadata": {
            "description": (
                "Base variant of Depth Anything V2 monocular depth estimation "
                "(MDE) model trained on synthetic labeled images and real "
                "unlabeled images."
            ),
            "params": 98_522_945,
            "path": "depth_anything",
        },
        "kaggle_handle": "kaggle://keras/depth-anything/keras/depth_anything_v2_base/1",
    },
    "depth_anything_v2_large": {
        "metadata": {
            "description": (
                "Large variant of Depth Anything V2 monocular depth estimation "
                "(MDE) model trained on synthetic labeled images and real "
                "unlabeled images."
            ),
            "params": 336_718_529,
            "path": "depth_anything",
        },
        "kaggle_handle": "kaggle://keras/depth-anything/keras/depth_anything_v2_large/1",
    },
}
