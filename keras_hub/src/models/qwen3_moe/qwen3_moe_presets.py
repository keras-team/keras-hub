"""Qwen3 MoE model preset configurations."""

backbone_presets = {
    "qwen3_moe_30b_a3b_en": {
        "metadata": {
            "description": (
                " Mixture-of-Experts (MoE) model has 30.5 billion total"
                " parameters with 3.3 billion activated, built on 48 layers"
                " and utilizes 32 query and 4 key/value attention heads"
                " with 128 experts (8 active)."
            ),
            "params": 30532122624,
            "path": "qwen3_moe",
        },
        "kaggle_handle": "kaggle://keras/qwen-3-moe/keras/qwen3_moe_30b_a3b_en/2",
    },
    "qwen3_moe_235b_a22b_en": {
        "metadata": {
            "description": (
                " Mixture-of-Experts (MoE) model has 235 billion"
                " total parameters with 22 billion activated, built on 94"
                " layers and utilizes 64 query and 4 key/value attention heads"
                " with 128 experts (8 active)."
            ),
            "params": 235093634560,
            "path": "qwen3_moe",
        },
        "kaggle_handle": "kaggle://keras/qwen-3-moe/keras/qwen3_moe_235b_a22b_en/1",
    },
}
