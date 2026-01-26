"""Qwen3-Omni model preset configurations."""

backbone_presets = {
    "qwen3_omni_30b_a3b_en": {
        "metadata": {
            "description": (
                "Qwen3-Omni Thinker (comprehension) model with 30.5 billion total "
                "parameters and 3.3 billion activated. This is a Mixture-of-Experts "
                "(MoE) based multimodal model supporting text, audio, image, and "
                "video inputs. Built on 48 layers with 32 query and 4 key/value "
                "attention heads, utilizing 128 experts (8 active per token). "
                "Features M-RoPE for multimodal position encoding."
            ),
            "params": 30532122624,
            "path": "qwen3_omni",
        },
        "kaggle_handle": "kaggle://keras/qwen-3-omni/keras/qwen3_omni_30b_a3b_en/1",
    },
}
