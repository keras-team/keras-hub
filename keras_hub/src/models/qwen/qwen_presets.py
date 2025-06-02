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
}
