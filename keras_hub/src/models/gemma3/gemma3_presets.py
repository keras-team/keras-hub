"""Gemma3 model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "gemma3_instruct_1b": {
        "metadata": {
            "description": (
                "1 billion parameter, 18-layer, text-only Gemma3 model."
            ),
            "params": 4299915632,
            "path": "gemma",
        },
        "kaggle_handle": "kaggle://kerashub/gemma3/keras/gemma3_instruct_4b/1",
    },
}
