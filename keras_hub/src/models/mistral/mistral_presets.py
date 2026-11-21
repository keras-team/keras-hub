"""Mistral model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    # Mistral models
    "mistral_7b_en": {
        "metadata": {
            "description": "Mistral 7B base model",
            "params": 7241732096,
            "path": "mistral",
        },
        "kaggle_handle": "kaggle://keras/mistral/keras/mistral_7b_en/8",
    },
    "mistral_0.3_7b_en": {
        "metadata": {
            "description": "Mistral 7B base version 0.3 model",
            "params": 7248023552,
            "path": "mistral",
        },
        "kaggle_handle": "kaggle://keras/mistral/keras/mistral_0.3_7b_en/1",
    },
    "mistral_instruct_7b_en": {
        "metadata": {
            "description": "Mistral 7B instruct model",
            "params": 7241732096,
            "path": "mistral",
        },
        "kaggle_handle": "kaggle://keras/mistral/keras/mistral_instruct_7b_en/8",
    },
    "mistral_0.2_instruct_7b_en": {
        "metadata": {
            "description": "Mistral 7B instruct version 0.2 model",
            "params": 7241732096,
            "path": "mistral",
        },
        "kaggle_handle": "kaggle://keras/mistral/keras/mistral_0.2_instruct_7b_en/3",
    },
    "mistral_0.3_instruct_7b_en": {
        "metadata": {
            "description": "Mistral 7B instruct version 0.3 model",
            "params": 7248023552,
            "path": "mistral",
        },
        "kaggle_handle": "kaggle://keras/mistral/keras/mistral_0.3_instruct_7b_en/1",
    },
    # Magistral models
    "magistral_small_2506_en": {
        "metadata": {
            "description": "Magistral Small 2506 model",
            "params": 23572403200,
            "path": "mistral",
        },
        "kaggle_handle": "kaggle://keras/magistral/keras/magistral_small_2506_en/1",
    },
    "magistral_small_2507_en": {
        "metadata": {
            "description": "Magistral Small 2507 model",
            "params": 23572403200,
            "path": "mistral",
        },
        "kaggle_handle": "kaggle://keras/magistral/keras/magistral_small_2507_en/1",
    },
}
