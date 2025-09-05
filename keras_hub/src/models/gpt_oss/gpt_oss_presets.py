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
        "config": {
            "vocabulary_size": 32000,
            "num_layers": 32,
            "num_query_heads": 32,
            "hidden_dim": 4096,
            "intermediate_dim": 14336,
            "num_key_value_heads": 8,
            "num_experts": 8,
            "top_k": 2,
            "rope_max_wavelength": 10000,
            "rope_scaling_factor": 1.0,
            "layer_norm_epsilon": 1e-6,
            "sliding_window": 4096,
            "dropout": 0.0,
            "use_bias": False,
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
        "config": {
            "vocabulary_size": 32000,
            "num_layers": 32,
            "num_query_heads": 32,
            "hidden_dim": 4096,
            "intermediate_dim": 14336,
            "num_key_value_heads": 8,
            "num_experts": 8,
            "top_k": 2,
            "rope_max_wavelength": 10000,
            "rope_scaling_factor": 1.0,
            "layer_norm_epsilon": 1e-6,
            "sliding_window": 4096,
            "dropout": 0.0,
            "use_bias": False,
        },
        "kaggle_handle": "kaggle://keras/gpt_oss/keras/gpt_oss_8_instruct_7b_en/1",
    },
}
