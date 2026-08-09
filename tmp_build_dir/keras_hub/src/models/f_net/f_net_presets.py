"""FNet model preset configurations."""

backbone_presets = {
    "f_net_base_en": {
        "metadata": {
            "description": (
                "12-layer FNet model where case is maintained. "
                "Trained on the C4 dataset."
            ),
            "params": 82861056,
            "path": "f_net",
        },
        "kaggle_handle": "kaggle://keras/f_net/keras/f_net_base_en/3",
    },
    "f_net_large_en": {
        "metadata": {
            "description": (
                "24-layer FNet model where case is maintained. "
                "Trained on the C4 dataset."
            ),
            "params": 236945408,
            "path": "f_net",
        },
        "kaggle_handle": "kaggle://keras/f_net/keras/f_net_large_en/3",
    },
}
