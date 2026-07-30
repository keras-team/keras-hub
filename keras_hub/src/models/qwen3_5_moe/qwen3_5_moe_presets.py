"""Qwen3.5 MoE model preset configurations."""

backbone_presets = {
    # Qwen3.5 MoE Model Presets
    "qwen3_5_moe_35b_a3b_base": {
        "metadata": {
            "description": (
                "35 billion total parameter Qwen3.5 MoE base model "
                "with ~3 billion active parameters per token. Features "
                "a 3:1 hybrid attention stack (GatedDeltaNet linear "
                "attention and full attention) with sparse "
                "Mixture-of-Experts feedforward for highly efficient "
                "inference. Supports text and multimodal inputs."
            ),
            "params": 35107181936,
            "path": "qwen3_5_moe",
        },
        "kaggle_handle": "kaggle://keras/qwen3-5-moe/keras/qwen3_5_moe_35b_a3b_base/1",
    },
    "qwen3_5_moe_35b_a3b": {
        "metadata": {
            "description": (
                "35 billion total parameter Qwen3.5 MoE instruction-tuned "
                "model with ~3 billion active parameters per token. "
                "Features a 3:1 hybrid attention stack (GatedDeltaNet "
                "linear attention and full attention) with sparse "
                "Mixture-of-Experts feedforward. Optimized for chat, "
                "reasoning, coding, and multimodal tasks."
            ),
            "params": 35107181936,
            "path": "qwen3_5_moe",
        },
        "kaggle_handle": "kaggle://keras/qwen3-5-moe/keras/qwen3_5_moe_35b_a3b/1",
    },
    # Qwen3.6 MoE Model Presets
    "qwen3_6_moe_35b_a3b": {
        "metadata": {
            "description": (
                "35 billion total parameter Qwen3.6 MoE instruction-tuned "
                "model with ~3 billion active parameters per token. "
                "Features a 3:1 hybrid attention stack (GatedDeltaNet "
                "linear attention and full attention) with sparse "
                "Mixture-of-Experts feedforward. Optimized for fast "
                "inference and extended context lengths."
            ),
            "params": 35107181936,
            "path": "qwen3_5_moe",
        },
        "kaggle_handle": "kaggle://keras/qwen3-6-moe/keras/qwen3_6_moe_35b_a3b/1",
    },
}
