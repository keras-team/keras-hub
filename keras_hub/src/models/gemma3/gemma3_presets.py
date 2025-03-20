"""Gemma model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "gemma3_instruct_4b": {
        "metadata": {
            "description": "4 billion parameter, 18-layer, Gemma model.",
            "params": 4299915632,
            "path": "gemma",
        },
        "kaggle_handle": "kaggle://kerashub/gemma3/keras/gemma3_instruct_4b/1",
    },
}
