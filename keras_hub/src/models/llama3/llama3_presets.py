"""Llama 3 model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "llama3_8b_en": {
        "metadata": {
            "description": "8 billion parameter, 32-layer, base LLaMA 3 model.",
            "params": 8030261248,
            "path": "llama3",
        },
        "kaggle_handle": "kaggle://keras/llama3/keras/llama3_8b_en/4",
    },
    "llama3_8b_en_int8": {
        "metadata": {
            "description": (
                "8 billion parameter, 32-layer, base LLaMA 3 model with "
                "activation and weights quantized to int8."
            ),
            "params": 8031894016,
            "path": "llama3",
        },
        "kaggle_handle": "kaggle://keras/llama3/keras/llama3_8b_en_int8/2",
    },
    "llama3_instruct_8b_en": {
        "metadata": {
            "description": (
                "8 billion parameter, 32-layer, instruction tuned LLaMA 3 "
                "model."
            ),
            "params": 8030261248,
            "path": "llama3",
        },
        "kaggle_handle": "kaggle://keras/llama3/keras/llama3_instruct_8b_en/4",
    },
    "llama3_instruct_8b_en_int8": {
        "metadata": {
            "description": (
                "8 billion parameter, 32-layer, instruction tuned LLaMA 3 "
                "model with activation and weights quantized to int8."
            ),
            "params": 8031894016,
            "path": "llama3",
        },
        "kaggle_handle": (
            "kaggle://keras/llama3/keras/llama3_instruct_8b_en_int8/2"
        ),
    },
}
