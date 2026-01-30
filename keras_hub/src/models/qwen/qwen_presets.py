"""Qwen preset configurations."""

backbone_presets = {
    "qwen2.5_0.5b_en": {
        "metadata": {
            "description": ("24-layer Qwen model with 0.5 billion parameters."),
            "params": 494032768,
            "path": "qwen",
        },
        "kaggle_handle": "kaggle://keras/qwen/keras/qwen2.5_0.5b_en/1",
    },
    "qwen2.5_3b_en": {
        "metadata": {
            "description": ("36-layer Qwen model with 3.1 billion parameters."),
            "params": 3085938688,
            "path": "qwen",
        },
        "kaggle_handle": "kaggle://keras/qwen/keras/qwen2.5_3b_en/1",
    },
    "qwen2.5_7b_en": {
        "metadata": {
            "description": ("48-layer Qwen model with 7 billion parameters."),
            "params": 6993420288,
            "path": "qwen",
        },
        "kaggle_handle": "kaggle://keras/qwen/keras/qwen2.5_7b_en/3",
    },
    "qwen2.5_instruct_0.5b_en": {
        "metadata": {
            "description": (
                "Instruction fine-tuned 24-layer Qwen model with 0.5 "
                "billion parameters."
            ),
            "params": 494032768,
            "path": "qwen",
        },
        "kaggle_handle": "kaggle://keras/qwen/keras/qwen2.5_instruct_0.5b_en/1",
    },
    "qwen2.5_instruct_32b_en": {
        "metadata": {
            "description": (
                "Instruction fine-tuned 64-layer Qwen model with 32 "
                "billion parameters."
            ),
            "params": 32763876352,
            "path": "qwen",
        },
        "kaggle_handle": "kaggle://keras/qwen/keras/qwen2.5_instruct_32b_en/2",
    },
    "qwen2.5_instruct_72b_en": {
        "metadata": {
            "description": (
                "Instruction fine-tuned 80-layer Qwen model with 72 "
                "billion parameters."
            ),
            "params": 72706203648,
            "path": "qwen",
        },
        "kaggle_handle": "kaggle://keras/qwen/keras/qwen2.5_instruct_72b_en/2",
    },
    "qwen2.5_coder_0.5b": {
        "metadata": {
            "description": (
                "Code-focused fine-tuned Qwen-2.5 model with 0.5 "
                "billion parameters."
            ),
            "params": 494032768,
            "path": "qwen",
        },
        "kaggle_handle": (
            "kaggle://keras/qwen2-5-coder/keras/qwen2.5_coder_0.5b/1"
        ),
    },
    "qwen2.5_coder_1.5b": {
        "metadata": {
            "description": (
                "Code-focused fine-tuned 28-layer Qwen-2.5 model with 1.5 "
                "billion parameters."
            ),
            "params": 1543434240,
            "path": "qwen",
        },
        "kaggle_handle": (
            "kaggle://keras/qwen2-5-coder/keras/qwen2.5_coder_1.5b/1"
        ),
    },
    "qwen2.5_coder_3b": {
        "metadata": {
            "description": (
                "Code-focused fine-tuned Qwen-2.5 model with 3 "
                "billion parameters."
            ),
            "params": 3085938688,
            "path": "qwen",
        },
        "kaggle_handle": (
            "kaggle://keras/qwen2-5-coder/keras/qwen2.5_coder_3b/1"
        ),
    },
    "qwen2.5_coder_7b": {
        "metadata": {
            "description": (
                "Code-focused fine-tuned Qwen-2.5 model with 7 "
                "billion parameters."
            ),
            "params": 6993420288,
            "path": "qwen",
        },
        "kaggle_handle": (
            "kaggle://keras/qwen2-5-coder/keras/qwen2.5_coder_7b/1"
        ),
    },
    "qwen2.5_coder_14b": {
        "metadata": {
            "description": (
                "Code-focused fine-tuned Qwen-2.5 model with 14 "
                "billion parameters."
            ),
            "params": 14000000000,
            "path": "qwen",
        },
        "kaggle_handle": (
            "kaggle://keras/qwen2-5-coder/keras/qwen2.5_coder_14b/1"
        ),
    },
    "qwen2.5_coder_32b": {
        "metadata": {
            "description": (
                "Code-focused fine-tuned Qwen-2.5 model with 32 "
                "billion parameters."
            ),
            "params": 32763876352,
            "path": "qwen",
        },
        "kaggle_handle": (
            "kaggle://keras/qwen2-5-coder/keras/qwen2.5_coder_32b/1"
        ),
    },
    "qwen2.5_coder_instruct_0.5b": {
        "metadata": {
            "description": (
                "Instruction-tuned code-focused Qwen-2.5 model with "
                "0.5 billion parameters."
            ),
            "params": 494032768,
            "path": "qwen",
        },
        "kaggle_handle": (
            "kaggle://keras/qwen2-5-coder/keras/qwen2.5_coder_instruct_0.5b/1"
        ),
    },
    "qwen2.5_coder_instruct_1.5b": {
        "metadata": {
            "description": (
                "Instruction-tuned code-focused Qwen-2.5 model with "
                "1.5 billion parameters."
            ),
            "params": 1543434240,
            "path": "qwen",
        },
        "kaggle_handle": (
            "kaggle://keras/qwen2-5-coder/keras/qwen2.5_coder_instruct_1.5b/1"
        ),
    },
    "qwen2.5_coder_instruct_3b": {
        "metadata": {
            "description": (
                "Instruction-tuned code-focused Qwen-2.5 model with "
                "3 billion parameters."
            ),
            "params": 3085938688,
            "path": "qwen",
        },
        "kaggle_handle": (
            "kaggle://keras/qwen2-5-coder/keras/qwen2.5_coder_instruct_3b/1"
        ),
    },
    "qwen2.5_coder_instruct_7b": {
        "metadata": {
            "description": (
                "Instruction-tuned code-focused Qwen-2.5 model with "
                "7 billion parameters."
            ),
            "params": 6993420288,
            "path": "qwen",
        },
        "kaggle_handle": (
            "kaggle://keras/qwen2-5-coder/keras/qwen2.5_coder_instruct_7b/1"
        ),
    },
    "qwen2.5_coder_instruct_14b": {
        "metadata": {
            "description": (
                "Instruction-tuned code-focused Qwen-2.5 model with "
                "14 billion parameters."
            ),
            "params": 14000000000,
            "path": "qwen",
        },
        "kaggle_handle": (
            "kaggle://keras/qwen2-5-coder/keras/qwen2.5_coder_instruct_14b/1"
        ),
    },
    "qwen2.5_coder_instruct_32b": {
        "metadata": {
            "description": (
                "Instruction-tuned code-focused Qwen-2.5 model with "
                "32 billion parameters."
            ),
            "params": 32763876352,
            "path": "qwen",
        },
        "kaggle_handle": (
            "kaggle://keras/qwen2-5-coder/keras/qwen2.5_coder_instruct_32b/1"
        ),
    },
}
