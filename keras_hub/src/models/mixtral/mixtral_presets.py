"""Mixtral preset configurations."""

backbone_presets = {
    "mixtral_8_7b_en": {
        "metadata": {
            "description": (
                "32-layer Mixtral MoE model with 7 billion"
                "active parameters and 8 experts per MoE layer."
            ),
            "params": 46702792704,
            "path": "mixtral",
        },
        "kaggle_handle": "kaggle://keras/mixtral/keras/mixtral_8_7b_en/4",
    },
    "mixtral_8_instruct_7b_en": {
        "metadata": {
            "description": (
                "Instruction fine-tuned 32-layer Mixtral MoE model"
                "with 7 billion active parameters and 8 experts per MoE layer."
            ),
            "params": 46702792704,
            "path": "mixtral",
        },
        "kaggle_handle": "kaggle://keras/mixtral/keras/mixtral_8_instruct_7b_en/4",
    },
}
