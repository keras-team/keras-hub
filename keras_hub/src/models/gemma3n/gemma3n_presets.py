"""Gemma3n model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "gemma3n_e2b": {
        "metadata": {
            "description": (
                "Gemma 3n E2B multimodal model "
                "(~5B total, ~2B effective parameters) supporting "
                "multimodal inputs and optimized for on-device deployment."
            ),
            "params": 5439595456,
            "path": "gemma3n",
        },
        "kaggle_handle": "kaggle://keras/gemma-3n/keras/gemma3n_e2b/1",
    },
    "gemma3n_e2b_it": {
        "metadata": {
            "description": (
                "Instruction-tuned Gemma 3n E2B multimodal model "
                "(~5B total, ~2B effective parameters) supporting "
                "multimodal inputs and optimized for on-device deployment."
            ),
            "params": 5439595456,
            "path": "gemma3n",
        },
        "kaggle_handle": "kaggle://keras/gemma-3n/keras/gemma3n_e2b_it/1",
    },
    "gemma3n_e4b": {
        "metadata": {
            "description": (
                "Gemma 3n E4B model with 8B total (4B effective) "
                "parameters, supporting multimodal inputs and "
                "optimized for on-device deployment."
            ),
            "params": 7850135376,
            "path": "gemma3n",
        },
        "kaggle_handle": "kaggle://keras/gemma-3n/keras/gemma3n_e4b/1",
    },
    "gemma3n_e4b_it": {
        "metadata": {
            "description": (
                "Instruction-tunedGemma 3n E4B model with 8B total "
                "(4B effective) parameters, supporting multimodal "
                "inputs and optimized for on-device deployment."
            ),
            "params": 7850135376,
            "path": "gemma3n",
        },
        "kaggle_handle": "kaggle://keras/gemma-3n/keras/gemma3n_e4b_it/1",
    },
}
