"""Qwen3-ASR model preset configurations."""

backbone_presets = {
    "qwen3_asr_1.7b": {
        "metadata": {
            "description": "Qwen3-ASR 1.7B model supporting 52 languages and dialects.",
            "params": 2000000000,
            "path": "qwen3_asr",
        },
        "kaggle_handle": "kaggle://keras/qwen3-asr/keras/qwen3_asr_1.7b/1",
    },
    "qwen3_asr_0.6b": {
        "metadata": {
            "description": "Qwen3-ASR 0.6B model supporting 52 languages and dialects.",
            "params": 900000000,
            "path": "qwen3_asr",
        },
        "kaggle_handle": "kaggle://keras/qwen3-asr/keras/qwen3_asr_0.6b/1",
    },
}
