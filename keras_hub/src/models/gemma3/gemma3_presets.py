"""Gemma model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "gemma_instruct_4b": {
        "metadata": {
            "description": "4 billion parameter, 18-layer, Gemma model.",
            "params": 2506172416,
            "path": "gemma",
        },
        "kaggle_handle": "kaggle://abheesht75/gemma3/keras/gemma3_instruct_4b/1",
    },
}
