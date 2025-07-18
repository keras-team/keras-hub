"""Qwen MoE preset configurations."""

backbone_presets = {
    "qwen1.5_moe_2.7b_en": {
        "metadata": {
            "description": (
                "24-layer Qwen MoE model with 2.7 billion active parameters "
                "and 8 experts per MoE layer."
            ),
            "params": 14315784192,
            "path": "qwen-1.5-moe",
        },
        "kaggle_handle": "kaggle://keras/qwen-1.5-moe/Keras/qwen1.5_moe_2.7b_en/4",
    },
}
