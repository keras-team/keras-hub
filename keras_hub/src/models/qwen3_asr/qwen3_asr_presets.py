"""Qwen3-ASR model preset configurations."""

backbone_presets = {
    "qwen3_asr_0.6b": {
        "metadata": {
            "description": (
                "Qwen3-ASR model with an 18-layer audio encoder and 0.6B "
                "parameters, optimized for multilingual speech recognition."
            ),
            "params": 782426112,
            "path": "qwen3_asr",
        },
        "kaggle_handle": "kaggle://keras/qwen3-asr/keras/qwen3_asr_0.6b/1",
    },
    "qwen3_asr_1.7b": {
        "metadata": {
            "description": (
                "Qwen3-ASR model with a 24-layer audio encoder and 1.7B "
                "parameters, offering high-quality multilingual speech "
                "recognition."
            ),
            "params": 2038052480,
            "path": "qwen3_asr",
        },
        "kaggle_handle": "kaggle://keras/qwen3-asr/keras/qwen3_asr_1.7b/1",
    },
}
