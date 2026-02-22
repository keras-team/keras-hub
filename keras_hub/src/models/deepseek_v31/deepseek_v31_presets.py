"""DeepSeek V3 model preset configurations."""

# Metadata for loading pretrained model weights and configurations.
backbone_presets = {
    "deepseek_v3_base": {
        "metadata": {
            "description": (
                "671 billion parameter, 61-layer, base DeepSeek V3 model. "
                "MoE architecture with 256 routed experts (8 per token). "
                "37B activated parameters."
            ),
            "params": 671000000000,
            "path": "deepseek_v3",
            "model_type": "MoE",
            "tokenizer": "DeepSeekV3Tokenizer",
        },
        "kaggle_handle": "kaggle://deepseek-ai/deepseek-v3/base/1",
    },
    "deepseek_v3": {
        "metadata": {
            "description": (
                "671 billion parameter, 61-layer, instruction-tuned "
                "DeepSeek V3 model. MoE architecture with 256 routed "
                "experts (8 per token). 37B activated parameters."
            ),
            "params": 671000000000,
            "path": "deepseek_v3",
            "model_type": "MoE",
            "tokenizer": "DeepSeekV3Tokenizer",
        },
        "kaggle_handle": "kaggle://deepseek-ai/deepseek-v3/instruct/1",
    },
}

# Tokenizer presets
tokenizer_presets = {
    "deepseek_v3_base": {
        "metadata": {
            "description": "DeepSeek V3 tokenizer.",
            "path": "deepseek_v3",
        },
        "kaggle_handle": "kaggle://deepseek-ai/deepseek-v3/tokenizer/1",
    },
    "deepseek_v3": {
        "metadata": {
            "description": "DeepSeek V3 tokenizer.",
            "path": "deepseek_v3",
        },
        "kaggle_handle": "kaggle://deepseek-ai/deepseek-v3/tokenizer/1",
    },
}

# Preprocessor presets
preprocessor_presets = {
    "deepseek_v3_base": {
        "metadata": {
            "description": "DeepSeek V3 preprocessor.",
            "path": "deepseek_v3",
        },
        "kaggle_handle": "kaggle://deepseek-ai/deepseek-v3/preprocessor/1",
    },
    "deepseek_v3": {
        "metadata": {
            "description": "DeepSeek V3 preprocessor.",
            "path": "deepseek_v3",
        },
        "kaggle_handle": "kaggle://deepseek-ai/deepseek-v3/preprocessor/1",
    },
}
