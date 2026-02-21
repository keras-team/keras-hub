"""DeepSeek V3.1 model preset configurations."""

# Metadata for loading pretrained model weights and configurations.
backbone_presets = {
    "deepseek_v3_1_base": {
        "metadata": {
            "description": (
                "671 billion parameter, 61-layer, base DeepSeek V3.1 model. "
                "MoE architecture with 256 routed experts (8 per token). "
                "37B activated parameters."
            ),
            "params": 671000000000,
            "path": "deepseek_v3_1",
            "model_type": "MoE",
            "tokenizer": "DeepSeekV3_1Tokenizer",
        },
        "kaggle_handle": "kaggle://keras/deepseek_v3_1/keras/deepseek_v3_1_base/1",
    },
    "deepseek_v3_1": {
        "metadata": {
            "description": (
                "671 billion parameter, 61-layer, instruction-tuned "
                "DeepSeek V3.1 model. MoE architecture with 256 routed "
                "experts (8 per token). 37B activated parameters."
            ),
            "params": 671000000000,
            "path": "deepseek_v3_1",
            "model_type": "MoE",
            "tokenizer": "DeepSeekV3_1Tokenizer",
        },
        "kaggle_handle": "kaggle://keras/deepseek_v3_1/keras/deepseek_v3_1/1",
    },
}

# Tokenizer presets
tokenizer_presets = {
    "deepseek_v3_1_base": {
        "metadata": {
            "description": "DeepSeek V3.1 tokenizer.",
            "path": "deepseek_v3_1",
        },
        "kaggle_handle": "kaggle://keras/deepseek_v3_1/keras/deepseek_v3_1_tokenizer/1",
    },
    "deepseek_v3_1": {
        "metadata": {
            "description": "DeepSeek V3.1 tokenizer.",
            "path": "deepseek_v3_1",
        },
        "kaggle_handle": "kaggle://keras/deepseek_v3_1/keras/deepseek_v3_1_tokenizer/1",
    },
}

# Preprocessor presets
preprocessor_presets = {
    "deepseek_v3_1_base": {
        "metadata": {
            "description": "DeepSeek V3.1 preprocessor.",
            "path": "deepseek_v3_1",
        },
        "kaggle_handle": "kaggle://keras/deepseek_v3_1/keras/deepseek_v3_1_preprocessor/1",
    },
    "deepseek_v3_1": {
        "metadata": {
            "description": "DeepSeek V3.1 preprocessor.",
            "path": "deepseek_v3_1",
        },
        "kaggle_handle": "kaggle://keras/deepseek_v3_1/keras/deepseek_v3_1_preprocessor/1",
    },
}
