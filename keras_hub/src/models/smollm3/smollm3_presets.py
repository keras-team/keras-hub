"""SmolLM3 model preset configurations."""

backbone_presets = {
    "smollm3_3b_en": {
        "metadata": {
            "description": (
                "Dense decoder-only model has 3 billion total parameters, "
                "built on 36 layers and utilizes 16 query and "
                "4 key/value attention heads."
            ),
            "params": 3075100928,
            "path": "smollm3",
        },
        "kaggle_handle": "kaggle://keras/smollm3/keras/smollm3_3b_en/1",
    },
}
