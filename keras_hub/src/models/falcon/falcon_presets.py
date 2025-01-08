"""Falcon model preset configurations."""

backbone_presets = {
    "falcon_refinedweb_1b_en": {
        "metadata": {
            "description": (
                "24-layer Falcon model (Falcon with 1B parameters), trained on "
                "350B tokens of RefinedWeb dataset."
            ),
            "params": 1311625216,
            "path": "falcon",
        },
        "kaggle_handle": "kaggle://keras/falcon/keras/falcon_refinedweb_1b_en/2",
    },
}
