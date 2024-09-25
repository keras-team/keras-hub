"""FNet model preset configurations."""

backbone_presets = {
    "f_net_base_en": {
        "metadata": {
            "description": (
                "12-layer FNet model where case is maintained. "
                "Trained on the C4 dataset."
            ),
            "params": 82861056,
            "official_name": "FNet",
            "path": "f_net",
            "model_card": "https://github.com/google-research/google-research/blob/master/f_net/README.md",
        },
        "kaggle_handle": "kaggle://keras/f_net/keras/f_net_base_en/2",
    },
    "f_net_large_en": {
        "metadata": {
            "description": (
                "24-layer FNet model where case is maintained. "
                "Trained on the C4 dataset."
            ),
            "params": 236945408,
            "official_name": "FNet",
            "path": "f_net",
            "model_card": "https://github.com/google-research/google-research/blob/master/f_net/README.md",
        },
        "kaggle_handle": "kaggle://keras/f_net/keras/f_net_large_en/2",
    },
}
