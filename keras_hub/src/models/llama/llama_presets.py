"""Llama model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "llama2_7b_en": {
        "metadata": {
            "description": "7 billion parameter, 32-layer, base LLaMA 2 model.",
            "params": 6738415616,
            "official_name": "LLaMA 2",
            "path": "llama2",
            "model_card": "https://github.com/meta-llama/llama",
        },
        "kaggle_handle": "kaggle://keras/llama2/keras/llama2_7b_en/1",
    },
    "llama2_7b_en_int8": {
        "metadata": {
            "description": (
                "7 billion parameter, 32-layer, base LLaMA 2 model with "
                "activation and weights quantized to int8."
            ),
            "params": 6739839488,
            "official_name": "LLaMA 2",
            "path": "llama2",
            "model_card": "https://github.com/meta-llama/llama",
        },
        "kaggle_handle": "kaggle://keras/llama2/keras/llama2_7b_en_int8/1",
    },
    "llama2_instruct_7b_en": {
        "metadata": {
            "description": (
                "7 billion parameter, 32-layer, instruction tuned LLaMA 2 "
                "model."
            ),
            "params": 6738415616,
            "official_name": "LLaMA 2",
            "path": "llama2",
            "model_card": "https://github.com/meta-llama/llama",
        },
        "kaggle_handle": "kaggle://keras/llama2/keras/llama2_instruct_7b_en/1",
    },
    "llama2_instruct_7b_en_int8": {
        "metadata": {
            "description": (
                "7 billion parameter, 32-layer, instruction tuned LLaMA 2 "
                "model with activation and weights quantized to int8."
            ),
            "params": 6739839488,
            "official_name": "LLaMA 2",
            "path": "llama2",
            "model_card": "https://github.com/meta-llama/llama",
        },
        "kaggle_handle": "kaggle://keras/llama2/keras/llama2_instruct_7b_en_int8/1",
    },
    "vicuna_1.5_7b_en": {
        "metadata": {
            "description": (
                "7 billion parameter, 32-layer, instruction tuned Vicuna v1.5 "
                "model."
            ),
            "params": 6738415616,
            "official_name": "Vicuna",
            "path": "vicuna",
            "model_card": "https://github.com/lm-sys/FastChat",
        },
        "kaggle_handle": "kaggle://keras/vicuna/keras/vicuna_1.5_7b_en/1",
    },
}
