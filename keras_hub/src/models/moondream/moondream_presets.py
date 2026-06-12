"""Moondream model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "moondream2": {
        "metadata": {
            "description": (
                "Moondream2: a tiny 1.87B parameter vision-language model "
                "designed to run efficiently on edge devices. Uses a SigLIP "
                "vision encoder (image size 378) and a Phi-1.5-style text "
                "decoder."
            ),
            "params": 1870000000,
            "official_name": "Moondream2",
            "path": "moondream",
            "model_card": "https://huggingface.co/vikhyatk/moondream2",
        },
        "kaggle_handle": "kaggle://keras/moondream/keras/moondream2/1",
    },
}
