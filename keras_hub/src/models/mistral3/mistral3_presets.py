"""Mistral3 model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    # Mistral Small models
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
    # Ministral 3 models
    "ministral_3_3b_base_2512_en": {
        "metadata": {
            "description": (
                "3 billion parameter, 26-layer, pretrained Ministral 3 "
                "model with a Pixtral vision encoder for image input."
            ),
            "params": 3849090048,
            "path": "mistral3",
        },
        "kaggle_handle": "kaggle://keras/ministral3/keras/ministral_3_3b_base_2512_en/1",
    },
    "ministral_3_3b_instruct_2512_en": {
        "metadata": {
            "description": (
                "3 billion parameter, 26-layer, instruction-tuned "
                "Ministral 3 model with a Pixtral vision encoder for image "
                "input."
            ),
            "params": 3849090048,
            "path": "mistral3",
        },
        "kaggle_handle": "kaggle://keras/ministral3/keras/ministral_3_3b_instruct_2512_en/1",
    },
    "ministral_3_3b_reasoning_2512_en": {
        "metadata": {
            "description": (
                "3 billion parameter, 26-layer, reasoning-tuned Ministral 3 "
                "model with a Pixtral vision encoder for image input."
            ),
            "params": 3849090048,
            "path": "mistral3",
        },
        "kaggle_handle": "kaggle://keras/ministral3/keras/ministral_3_3b_reasoning_2512_en/1",
    },
    "ministral_3_8b_base_2512_en": {
        "metadata": {
            "description": (
                "8 billion parameter, 34-layer, pretrained Ministral 3 "
                "model with a Pixtral vision encoder for image input."
            ),
            "params": 8918026240,
            "path": "mistral3",
        },
        "kaggle_handle": "kaggle://keras/ministral3/keras/ministral_3_8b_base_2512_en/1",
    },
    "ministral_3_8b_instruct_2512_en": {
        "metadata": {
            "description": (
                "8 billion parameter, 34-layer, instruction-tuned "
                "Ministral 3 model with a Pixtral vision encoder for image "
                "input."
            ),
            "params": 8918026240,
            "path": "mistral3",
        },
        "kaggle_handle": "kaggle://keras/ministral3/keras/ministral_3_8b_instruct_2512_en/1",
    },
    "ministral_3_8b_reasoning_2512_en": {
        "metadata": {
            "description": (
                "8 billion parameter, 34-layer, reasoning-tuned Ministral 3 "
                "model with a Pixtral vision encoder for image input."
            ),
            "params": 8918026240,
            "path": "mistral3",
        },
        "kaggle_handle": "kaggle://keras/ministral3/keras/ministral_3_8b_reasoning_2512_en/1",
    },
    "ministral_3_14b_base_2512_en": {
        "metadata": {
            "description": (
                "14 billion parameter, 40-layer, pretrained Ministral 3 "
                "model with a Pixtral vision encoder for image input."
            ),
            "params": 13945031680,
            "path": "mistral3",
        },
        "kaggle_handle": "kaggle://keras/ministral3/keras/ministral_3_14b_base_2512_en/1",
    },
    "ministral_3_14b_instruct_2512_en": {
        "metadata": {
            "description": (
                "14 billion parameter, 40-layer, instruction-tuned "
                "Ministral 3 model with a Pixtral vision encoder for image "
                "input."
            ),
            "params": 13945031680,
            "path": "mistral3",
        },
        "kaggle_handle": "kaggle://keras/ministral3/keras/ministral_3_14b_instruct_2512_en/1",
    },
    "ministral_3_14b_reasoning_2512_en": {
        "metadata": {
            "description": (
                "14 billion parameter, 40-layer, reasoning-tuned Ministral 3 "
                "model with a Pixtral vision encoder for image input."
            ),
            "params": 13945031680,
            "path": "mistral3",
        },
        "kaggle_handle": "kaggle://keras/ministral3/keras/ministral_3_14b_reasoning_2512_en/1",
    },
    # Shieldstral models
    "shieldstral_1.0_3b_en": {
        "metadata": {
            "description": (
                "3 billion parameter, 26-layer Shieldstral model with a "
                "Pixtral vision encoder. The model checks text and image "
                "content against a safety policy written in natural "
                "language. The model returns one safety score for each "
                "input."
            ),
            "params": 3849090048,
            "path": "mistral3",
        },
        "kaggle_handle": "kaggle://keras/shieldstral/keras/shieldstral_1.0_3b_en/1",
    },
    # Devstral 2 models
    "devstral_small_2_24b_instruct_2512_en": {
        "metadata": {
            "description": (
                "24 billion parameter, 40-layer, instruction-tuned Devstral "
                "Small 2 model for agentic software engineering, with a "
                "Pixtral vision encoder for image input."
            ),
            "params": 24011361280,
            "path": "mistral3",
        },
        "kaggle_handle": "kaggle://keras/devstral2/keras/devstral_small_2_24b_instruct_2512_en/1",
    },
    # Magistral models
    "magistral_small_2509_en": {
        "metadata": {
            "description": (
                "24 billion parameter, 40-layer, reasoning-tuned Magistral "
                "Small model with a Pixtral vision encoder for image input."
            ),
            "params": 24011361280,
            "path": "mistral3",
        },
        "kaggle_handle": "kaggle://keras/magistral/keras/magistral_small_2509_en/1",
    },
}
