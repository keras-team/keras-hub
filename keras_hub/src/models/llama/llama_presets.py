"""Llama model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "llama2_7b_en": {
        "metadata": {
            "description": "7 billion parameter, 32-layer, base LLaMA 2 model.",
            "params": 6738415616,
            "path": "llama",
        },
        "kaggle_handle": "kaggle://keras/llama2/keras/llama2_7b_en/3",
    },
    "llama2_7b_en_int8": {
        "metadata": {
            "description": (
                "7 billion parameter, 32-layer, base LLaMA 2 model with "
                "activation and weights quantized to int8."
            ),
            "params": 6739839488,
            "path": "llama",
        },
        "kaggle_handle": "kaggle://keras/llama2/keras/llama2_7b_en_int8/2",
    },
    "llama2_instruct_7b_en": {
        "metadata": {
            "description": (
                "7 billion parameter, 32-layer, instruction tuned LLaMA 2 "
                "model."
            ),
            "params": 6738415616,
            "path": "llama",
        },
        "kaggle_handle": "kaggle://keras/llama2/keras/llama2_instruct_7b_en/3",
    },
    "llama2_instruct_7b_en_int8": {
        "metadata": {
            "description": (
                "7 billion parameter, 32-layer, instruction tuned LLaMA 2 "
                "model with activation and weights quantized to int8."
            ),
            "params": 6739839488,
            "path": "llama",
        },
        "kaggle_handle": "kaggle://keras/llama2/keras/llama2_instruct_7b_en_int8/2",
    },
    "vicuna_1.5_7b_en": {
        "metadata": {
            "description": (
                "7 billion parameter, 32-layer, instruction tuned Vicuna v1.5 "
                "model."
            ),
            "params": 6738415616,
            "path": "llama",
        },
        "kaggle_handle": "kaggle://keras/vicuna/keras/vicuna_1.5_7b_en/3",
    },
}
