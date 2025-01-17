"""Mistral model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "mistral_7b_en": {
        "metadata": {
            "description": "Mistral 7B base model",
            "params": 7241732096,
            "path": "mistral",
        },
        "kaggle_handle": "kaggle://keras/mistral/keras/mistral_7b_en/7",
    },
    "mistral_instruct_7b_en": {
        "metadata": {
            "description": "Mistral 7B instruct model",
            "params": 7241732096,
            "path": "mistral",
        },
        "kaggle_handle": "kaggle://keras/mistral/keras/mistral_instruct_7b_en/7",
    },
    "mistral_0.2_instruct_7b_en": {
        "metadata": {
            "description": "Mistral 7B instruct Version 0.2 model",
            "params": 7241732096,
            "path": "mistral",
        },
        "kaggle_handle": "kaggle://keras/mistral/keras/mistral_0.2_instruct_7b_en/2",
    },
}
