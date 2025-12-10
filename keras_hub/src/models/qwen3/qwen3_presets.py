"""Qwen3 model preset configurations."""

backbone_presets = {
    "qwen3_0.6b_en": {
        "metadata": {
            "description": (
                "28-layer Qwen3 model with 596M parameters, optimized for "
                "efficiency and fast inference on resource-constrained devices."
            ),
            "params": 596049920,
            "path": "qwen3",
        },
        "kaggle_handle": "kaggle://keras/qwen-3/keras/qwen3_0.6b_en/1",
    },
    "qwen3_1.7b_en": {
        "metadata": {
            "description": (
                "28-layer Qwen3 model with 1.72B parameters, offering "
                "a good balance between performance and resource usage."
            ),
            "params": 1720574976,
            "path": "qwen3",
        },
        "kaggle_handle": "kaggle://keras/qwen-3/keras/qwen3_1.7b_en/1",
    },
    "qwen3_4b_en": {
        "metadata": {
            "description": (
                "36-layer Qwen3 model with 4.02B parameters, offering improved "
                "reasoning capabilities and better performance than smaller "
                "variants."
            ),
            "params": 4022468096,
            "path": "qwen3",
        },
        "kaggle_handle": "kaggle://keras/qwen-3/keras/qwen3_4b_en/1",
    },
    "qwen3_8b_en": {
        "metadata": {
            "description": (
                "36-layer Qwen3 model with 8.19B parameters, featuring "
                "enhanced reasoning, coding, and instruction-following "
                "capabilities."
            ),
            "params": 8190735360,
            "path": "qwen3",
        },
        "kaggle_handle": "kaggle://keras/qwen-3/keras/qwen3_8b_en/1",
    },
    "qwen3_14b_en": {
        "metadata": {
            "description": (
                "40-layer Qwen3 model with 14.77B parameters, featuring "
                "advanced reasoning, coding, and multilingual capabilities."
            ),
            "params": 14768307200,
            "path": "qwen3",
        },
        "kaggle_handle": "kaggle://keras/qwen-3/keras/qwen3_14b_en/1",
    },
    "qwen3_32b_en": {
        "metadata": {
            "description": (
                "64-layer Qwen3 model with 32.76B parameters, featuring "
                "state-of-the-art performance across reasoning, coding, and "
                "general language tasks."
            ),
            "params": 32762123264,
            "path": "qwen3",
        },
        "kaggle_handle": "kaggle://keras/qwen-3/keras/qwen3_32b_en/1",
    },
    "qwen3_embedding_0.6b_en": {
        "metadata": {
            "description": (
                "This text embedding model features a 32k context length and "
                "offers flexible, user-defined embedding dimensions that can "
                "range from 32 to 1024."
            ),
            "params": 595776512,
            "path": "qwen3",
        },
        "kaggle_handle": "kaggle://keras/qwen-3-embedding/keras/qwen3_embedding_0.6b_en/1",
    },
    "qwen3_embedding_4b_en": {
        "metadata": {
            "description": (
                "This text embedding model features a 32k context length and "
                "offers flexible, user-defined embedding dimensions that can "
                "range from 32 to 2560."
            ),
            "params": 4021774336,
            "path": "qwen3",
        },
        "kaggle_handle": "kaggle://keras/qwen-3-embedding/keras/qwen3_embedding_4b_en/1",
    },
    "qwen3_embedding_8b_en": {
        "metadata": {
            "description": (
                "This text embedding model features a 32k context length and "
                "offers flexible, user-defined embedding dimensions that can "
                "range from 32 to 4096."
            ),
            "params": 8188515328,
            "path": "qwen3",
        },
        "kaggle_handle": "kaggle://keras/qwen-3-embedding/keras/qwen3_embedding_8b_en/1",
    },
}
