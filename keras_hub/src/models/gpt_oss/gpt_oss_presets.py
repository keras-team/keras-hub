"""GPT-OSS preset configurations."""

backbone_presets = {
    "gpt_oss_8_7b_en": {
        "metadata": {
            "description": (
                "32-layer GPT-OSS MoE model with 7 billion"
                "active parameters and 8 experts per MoE layer."
            ),
            "params": 46702792704,  # Total parameters, similar to Mixtral 8x7B
            "path": "gpt_oss",
        },
        "kaggle_handle": "kaggle://keras/gpt_oss/keras/gpt_oss_8_7b_en/1",
    },
    "gpt_oss_8_instruct_7b_en": {
        "metadata": {
            "description": (
                "Instruction fine-tuned 32-layer GPT-OSS MoE model"
                "with 7 billion active parameters and 8 experts per MoE layer."
            ),
            "params": 46702792704,  # Total parameters, similar to Mixtral 8x7B
            "path": "gpt_oss",
        },
        "kaggle_handle": "kaggle://keras/gpt_oss/keras/gpt_oss_8_instruct_7b_en/1",
    },
}
