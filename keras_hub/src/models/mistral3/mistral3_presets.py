"""Mistral3 model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "mistral_small_3.1_24b_base_2503_en": {
        "metadata": {
            "description": (
                "24 billion parameter, 40-layer, pretrained Mistral3 model "
                "with a Pixtral vision encoder for image input."
            ),
            "params": 24011361280,
            "path": "mistral3",
        },
        "kaggle_handle": "kaggle://keras/mistral3/keras/mistral_small_3.1_24b_base_2503_en/1",
    },
    "mistral_small_3.1_24b_instruct_2503_en": {
        "metadata": {
            "description": (
                "24 billion parameter, 40-layer, instruction-tuned "
                "Mistral3 model with a Pixtral vision encoder for image "
                "input."
            ),
            "params": 24011361280,
            "path": "mistral3",
        },
        "kaggle_handle": "kaggle://keras/mistral3/keras/mistral_small_3.1_24b_instruct_2503_en/1",
    },
    "mistral_small_3.2_24b_instruct_2506_en": {
        "metadata": {
            "description": (
                "24 billion parameter, 40-layer, instruction-tuned "
                "Mistral3 model with a Pixtral vision encoder for image "
                "input. An updated version of "
                "mistral_small_3.1_24b_instruct_2503_en with improved "
                "instruction-following and reduced repetition."
            ),
            "params": 24011361280,
            "path": "mistral3",
        },
        "kaggle_handle": "kaggle://keras/mistral3/keras/mistral_small_3.2_24b_instruct_2506_en/1",
    },
}
