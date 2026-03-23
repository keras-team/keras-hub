# Metadata for loading pretrained model weights.
backbone_presets = {
    "qwen3_asr_0.6b": {
        "metadata": {
            "description": (
                "Qwen3-ASR 0.6B model for multilingual speech recognition. "
                "Combines an 18-layer audio encoder with a 28-layer Qwen3 "
                "decoder. Supports 30 languages and 22 Chinese dialects."
            ),
            "params": 600000000,
            "path": "qwen3_asr",
        },
        "kaggle_handle": ("kaggle://keras/qwen3-asr/keras/qwen3_asr_0.6b/1"),
    },
    "qwen3_asr_1.7b": {
        "metadata": {
            "description": (
                "Qwen3-ASR 1.7B model for multilingual speech recognition. "
                "Combines a 24-layer audio encoder with a 28-layer Qwen3 "
                "decoder. Supports 30 languages and 22 Chinese dialects."
            ),
            "params": 1700000000,
            "path": "qwen3_asr",
        },
        "kaggle_handle": ("kaggle://keras/qwen3-asr/keras/qwen3_asr_1.7b/1"),
    },
}
