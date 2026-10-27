"""BLIP-2 model preset configurations."""

backbone_presets = {
    "blip2_opt_2.7b": {
        "metadata": {
            "description": (
                "BLIP-2 model using OPT-2.7B as the frozen language model."
            ),
            "params": 3744761856,
            "path": "blip2",
        },
        "kaggle_handle": "kaggle://keras/blip2/keras/blip2_opt_2.7b/1",
    },
    "blip2_opt_6.7b": {
        "metadata": {
            "description": (
                "BLIP-2 model using OPT-6.7B as the frozen language model."
            ),
            "params": 7752869376,
            "path": "blip2",
        },
        "kaggle_handle": "kaggle://keras/blip2/keras/blip2_opt_6.7b/1",
    },
    "blip2_flan_t5_xl": {
        "metadata": {
            "description": (
                "BLIP-2 model using Flan-T5-XL (~3B) as the frozen "
                "language model."
            ),
            "params": 3942446592,
            "path": "blip2",
        },
        "kaggle_handle": "kaggle://keras/blip2/keras/blip2_flan_t5_xl/1",
    },
    "blip2_flan_t5_xxl": {
        "metadata": {
            "description": (
                "BLIP-2 model using Flan-T5-XXL (~11B) as the frozen "
                "language model."
            ),
            "params": 12229596672,
            "path": "blip2",
        },
        "kaggle_handle": "kaggle://keras/blip2/keras/blip2_flan_t5_xxl/1",
    },
}
