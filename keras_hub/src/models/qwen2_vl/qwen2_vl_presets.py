"""Qwen2-VL preset configurations."""

backbone_presets = {
    "qwen2_vl_2b_instruct": {
        "metadata": {
            "description": (
                "28-layer Qwen2-VL multimodal model with 2 billion "
                "parameters, instruction-tuned."
            ),
            "params": 2210000000,
            "path": "qwen2_vl",
        },
        "kaggle_handle": (
            "kaggle://keras/qwen2-vl/keras/qwen2_vl_2b_instruct/1"
        ),
    },
    "qwen2_vl_7b_instruct": {
        "metadata": {
            "description": (
                "28-layer Qwen2-VL multimodal model with 7 billion "
                "parameters, instruction-tuned."
            ),
            "params": 8290000000,
            "path": "qwen2_vl",
        },
        "kaggle_handle": (
            "kaggle://keras/qwen2-vl/keras/qwen2_vl_7b_instruct/1"
        ),
    },
    "qwen2_vl_72b_instruct": {
        "metadata": {
            "description": (
                "80-layer Qwen2-VL multimodal model with 72 billion "
                "parameters, instruction-tuned."
            ),
            "params": 73400000000,
            "path": "qwen2_vl",
        },
        "kaggle_handle": (
            "kaggle://keras/qwen2-vl/keras/qwen2_vl_72b_instruct/1"
        ),
    },
}
