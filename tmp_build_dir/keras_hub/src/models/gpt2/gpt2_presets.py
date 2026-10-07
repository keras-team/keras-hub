"""GPT-2 model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "gpt2_base_en": {
        "metadata": {
            "description": (
                "12-layer GPT-2 model where case is maintained. "
                "Trained on WebText."
            ),
            "params": 124439808,
            "path": "gpt2",
        },
        "kaggle_handle": "kaggle://keras/gpt2/keras/gpt2_base_en/3",
    },
    "gpt2_medium_en": {
        "metadata": {
            "description": (
                "24-layer GPT-2 model where case is maintained. "
                "Trained on WebText."
            ),
            "params": 354823168,
            "path": "gpt2",
        },
        "kaggle_handle": "kaggle://keras/gpt2/keras/gpt2_medium_en/3",
    },
    "gpt2_large_en": {
        "metadata": {
            "description": (
                "36-layer GPT-2 model where case is maintained. "
                "Trained on WebText."
            ),
            "params": 774030080,
            "path": "gpt2",
        },
        "kaggle_handle": "kaggle://keras/gpt2/keras/gpt2_large_en/3",
    },
    "gpt2_extra_large_en": {
        "metadata": {
            "description": (
                "48-layer GPT-2 model where case is maintained. "
                "Trained on WebText."
            ),
            "params": 1557611200,
            "path": "gpt2",
        },
        "kaggle_handle": "kaggle://keras/gpt2/keras/gpt2_extra_large_en/3",
    },
    "gpt2_base_en_cnn_dailymail": {
        "metadata": {
            "description": (
                "12-layer GPT-2 model where case is maintained. "
                "Finetuned on the CNN/DailyMail summarization dataset."
            ),
            "params": 124439808,
            "path": "gpt2",
        },
        "kaggle_handle": "kaggle://keras/gpt2/keras/gpt2_base_en_cnn_dailymail/3",
    },
}
