"""Gemma model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "gemma_2b_en": {
        "metadata": {
            "description": "2 billion parameter, 18-layer, base Gemma model.",
            "params": 2506172416,
            "path": "gemma",
        },
        "kaggle_handle": "kaggle://keras/gemma/keras/gemma_2b_en/3",
    },
    "gemma_instruct_2b_en": {
        "metadata": {
            "description": (
                "2 billion parameter, 18-layer, instruction tuned Gemma model."
            ),
            "params": 2506172416,
            "path": "gemma",
        },
        "kaggle_handle": "kaggle://keras/gemma/keras/gemma_instruct_2b_en/3",
    },
    "gemma_1.1_instruct_2b_en": {
        "metadata": {
            "description": (
                "2 billion parameter, 18-layer, instruction tuned Gemma model. "
                "The 1.1 update improves model quality."
            ),
            "params": 2506172416,
            "path": "gemma",
        },
        "kaggle_handle": "kaggle://keras/gemma/keras/gemma_1.1_instruct_2b_en/4",
    },
    "code_gemma_1.1_2b_en": {
        "metadata": {
            "description": (
                "2 billion parameter, 18-layer, CodeGemma model. This model "
                "has been trained on a fill-in-the-middle (FIM) task for code "
                "completion. The 1.1 update improves model quality."
            ),
            "params": 2506172416,
            "path": "gemma",
        },
        "kaggle_handle": "kaggle://keras/codegemma/keras/code_gemma_1.1_2b_en/2",
    },
    "code_gemma_2b_en": {
        "metadata": {
            "description": (
                "2 billion parameter, 18-layer, CodeGemma model. This model "
                "has been trained on a fill-in-the-middle (FIM) task for code "
                "completion."
            ),
            "params": 2506172416,
            "path": "gemma",
        },
        "kaggle_handle": "kaggle://keras/codegemma/keras/code_gemma_2b_en/2",
    },
    "gemma_7b_en": {
        "metadata": {
            "description": "7 billion parameter, 28-layer, base Gemma model.",
            "params": 8537680896,
            "path": "gemma",
        },
        "kaggle_handle": "kaggle://keras/gemma/keras/gemma_7b_en/4",
    },
    "gemma_instruct_7b_en": {
        "metadata": {
            "description": (
                "7 billion parameter, 28-layer, instruction tuned Gemma model."
            ),
            "params": 8537680896,
            "path": "gemma",
        },
        "kaggle_handle": "kaggle://keras/gemma/keras/gemma_instruct_7b_en/4",
    },
    "gemma_1.1_instruct_7b_en": {
        "metadata": {
            "description": (
                "7 billion parameter, 28-layer, instruction tuned Gemma model. "
                "The 1.1 update improves model quality."
            ),
            "params": 8537680896,
            "path": "gemma",
        },
        "kaggle_handle": "kaggle://keras/gemma/keras/gemma_1.1_instruct_7b_en/5",
    },
    "code_gemma_7b_en": {
        "metadata": {
            "description": (
                "7 billion parameter, 28-layer, CodeGemma model. This model "
                "has been trained on a fill-in-the-middle (FIM) task for code "
                "completion."
            ),
            "params": 8537680896,
            "path": "gemma",
        },
        "kaggle_handle": "kaggle://keras/codegemma/keras/code_gemma_7b_en/3",
    },
    "code_gemma_instruct_7b_en": {
        "metadata": {
            "description": (
                "7 billion parameter, 28-layer, instruction tuned CodeGemma "
                "model. This model has been trained for chat use cases related "
                "to code."
            ),
            "params": 8537680896,
            "path": "gemma",
        },
        "kaggle_handle": "kaggle://keras/codegemma/keras/code_gemma_instruct_7b_en/3",
    },
    "code_gemma_1.1_instruct_7b_en": {
        "metadata": {
            "description": (
                "7 billion parameter, 28-layer, instruction tuned CodeGemma "
                "model. This model has been trained for chat use cases related "
                "to code. The 1.1 update improves model quality."
            ),
            "params": 8537680896,
            "path": "gemma",
        },
        "kaggle_handle": "kaggle://keras/codegemma/keras/code_gemma_1.1_instruct_7b_en/3",
    },
    "gemma2_2b_en": {
        "metadata": {
            "description": "2 billion parameter, 26-layer, base Gemma model.",
            "params": 2614341888,
            "path": "gemma",
        },
        "kaggle_handle": "kaggle://keras/gemma2/keras/gemma2_2b_en/2",
    },
    "gemma2_instruct_2b_en": {
        "metadata": {
            "description": (
                "2 billion parameter, 26-layer, instruction tuned Gemma model."
            ),
            "params": 2614341888,
            "path": "gemma",
        },
        "kaggle_handle": "kaggle://keras/gemma2/keras/gemma2_instruct_2b_en/2",
    },
    "gemma2_9b_en": {
        "metadata": {
            "description": "9 billion parameter, 42-layer, base Gemma model.",
            "params": 9241705984,
            "path": "gemma",
        },
        "kaggle_handle": "kaggle://keras/gemma2/keras/gemma2_9b_en/4",
    },
    "gemma2_instruct_9b_en": {
        "metadata": {
            "description": (
                "9 billion parameter, 42-layer, instruction tuned Gemma model."
            ),
            "params": 9241705984,
            "path": "gemma",
        },
        "kaggle_handle": "kaggle://keras/gemma2/keras/gemma2_instruct_9b_en/4",
    },
    "gemma2_27b_en": {
        "metadata": {
            "description": "27 billion parameter, 42-layer, base Gemma model.",
            "params": 27227128320,
            "path": "gemma",
        },
        "kaggle_handle": "kaggle://keras/gemma2/keras/gemma2_27b_en/3",
    },
    "gemma2_instruct_27b_en": {
        "metadata": {
            "description": (
                "27 billion parameter, 42-layer, instruction tuned Gemma model."
            ),
            "params": 27227128320,
            "path": "gemma",
        },
        "kaggle_handle": "kaggle://keras/gemma2/keras/gemma2_instruct_27b_en/3",
    },
    "shieldgemma_2b_en": {
        "metadata": {
            "description": "2 billion parameter, 26-layer, ShieldGemma model.",
            "params": 2614341888,
            "path": "gemma",
        },
        "kaggle_handle": "kaggle://google/shieldgemma/keras/shieldgemma_2b_en/2",
    },
    "shieldgemma_9b_en": {
        "metadata": {
            "description": "9 billion parameter, 42-layer, ShieldGemma model.",
            "params": 9241705984,
            "path": "gemma",
        },
        "kaggle_handle": "kaggle://google/shieldgemma/keras/shieldgemma_9b_en/2",
    },
    "shieldgemma_27b_en": {
        "metadata": {
            "description": "27 billion parameter, 42-layer, ShieldGemma model.",
            "params": 27227128320,
            "path": "gemma",
        },
        "kaggle_handle": "kaggle://google/shieldgemma/keras/shieldgemma_27b_en/2",
    },
}
