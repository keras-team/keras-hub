"""Qwen3-ASR model preset configurations."""

backbone_presets = {
    "qwen3_asr_0.6b": {
        "metadata": {
            "description": (
                "18-layer Qwen3-ASR model with 0.6B parameters, optimized for "
                "multilingual speech recognition."
            ),
            "params": 782437760,
            "path": "qwen3_asr",
        },
        "kaggle_handle": "kaggle://keras/qwen3-asr/keras/qwen3_asr_0.6b/1",
    },
    "qwen3_asr_1.7b": {
        "metadata": {
            "description": (
                "24-layer Qwen3-ASR model with 1.7B parameters, offering "
                "high-quality multilingual speech recognition."
            ),
            "params": 2038065792,
            "path": "qwen3_asr",
        },
        "kaggle_handle": "kaggle://keras/qwen3-asr/keras/qwen3_asr_1.7b/1",
    },
}
