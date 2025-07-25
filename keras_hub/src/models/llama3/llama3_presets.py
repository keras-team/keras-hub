"""Llama 3 model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "llama3_8b_en": {
        "metadata": {
            "description": "8 billion parameter, 32-layer, base LLaMA 3 model.",
            "params": 8030261248,
            "path": "llama3",
        },
        "kaggle_handle": "kaggle://keras/llama3/keras/llama3_8b_en/5",
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
        "kaggle_handle": "kaggle://keras/llama3/keras/llama3_instruct_8b_en/5",
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
    "llama3.1_8b": {
        "metadata": {
            "description": (
                "8 billion parameter, 32-layer, based LLaMA 3.1 model. "
            ),
            "params": 8030261248,
            "path": "llama3",
        },
        "kaggle_handle": ("kaggle://keras/llama3/keras/llama3.1_8b/2"),
    },
    "llama3.1_instruct_8b": {
        "metadata": {
            "description": (
                "8 billion parameter, 32-layer, instruction tuned LLaMA 3.1. "
            ),
            "params": 8030261248,
            "path": "llama3",
        },
        "kaggle_handle": ("kaggle://keras/llama3/keras/lama3.1_instruct_8b/2"),
    },
    "llama3.1_guard_8b": {
        "metadata": {
            "description": (
                "8 billion parameter, 32-layer, LLaMA 3.1 fine-tuned for "
                "consent safety classification. "
            ),
            "params": 8030261248,
            "path": "llama3",
        },
        "kaggle_handle": ("kaggle://keras/llama3/keras/llama3.1_guard_8b/2"),
    },
    "llama3.2_1b": {
        "metadata": {
            "description": (
                "1 billion parameter, 16-layer, based LLaMA 3.2 model. "
            ),
            "params": 1498482688,
            "path": "llama3",
        },
        "kaggle_handle": ("kaggle://keras/llama3/keras/llama3.2_1b/1"),
    },
    "llama3.2_instruct_1b": {
        "metadata": {
            "description": (
                "1 billion parameter, 16-layer, instruction tuned LLaMA 3.2. "
            ),
            "params": 1498482688,
            "path": "llama3",
        },
        "kaggle_handle": ("kaggle://keras/llama3/keras/llama3.2_instruct_1b/1"),
    },
    "llama3.2_3b": {
        "metadata": {
            "description": (
                "3 billion parameter, 26-layer, based LLaMA 3.2 model. "
            ),
            "params": 3606752256,
            "path": "llama3",
        },
        "kaggle_handle": ("kaggle://keras/llama3/keras/llama3.2_3b/1"),
    },
    "llama3.2_instruct_3b": {
        "metadata": {
            "description": (
                "3 billion parameter, 28-layer, instruction tuned LLaMA 3.2. "
            ),
            "params": 3606752256,
            "path": "llama3",
        },
        "kaggle_handle": ("kaggle://keras/llama3/keras/llama3.2_instruct_3b/1"),
    },
    "llama3.2_guard_1b": {
        "metadata": {
            "description": (
                "1 billion parameter, 16-layer, based LLaMA 3.2 model "
                "fine-tuned for consent safety classification. "
            ),
            "params": 1498482688,
            "path": "llama3",
        },
        "kaggle_handle": ("kaggle://keras/llama3/keras/llama3.2_guard_1b/1"),
    },
}
