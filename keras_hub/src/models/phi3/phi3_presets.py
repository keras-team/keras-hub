"""Phi-3 model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "phi3_mini_4k_instruct_en": {
        "metadata": {
            "description": (
                "3.8 billion parameters, 32 layers, 4k context length, Phi-3 "
                "model. The model was trained using the Phi-3 datasets. This "
                "dataset includes both synthetic data and filtered publicly "
                "available website data, with an emphasis on high-quality and "
                "reasoning-dense properties."
            ),
            "params": 3821079552,
            "path": "phi3",
        },
        "kaggle_handle": "kaggle://keras/phi3/keras/phi3_mini_4k_instruct_en/2",
    },
    "phi3_mini_128k_instruct_en": {
        "metadata": {
            "description": (
                "3.8 billion parameters, 32 layers, 128k context length, Phi-3 "
                "model. The model was trained using the Phi-3 datasets. This "
                "dataset includes both synthetic data and filtered publicly "
                "available website data, with an emphasis on high-quality and "
                "reasoning-dense properties."
            ),
            "params": 3821079552,
            "path": "phi3",
        },
        "kaggle_handle": "kaggle://keras/phi3/keras/phi3_mini_128k_instruct_en/2",
    },
}
