"""T5 model preset configurations."""

backbone_presets = {
    "t5_small_multi": {
        "metadata": {
            "description": (
                "8-layer T5 model. Trained on the Colossal Clean Crawled "
                "Corpus (C4)."
            ),
            "params": 0,
            "official_name": "T5",
            "path": "t5",
            "model_card": "https://github.com/google-research/text-to-text-transfer-transformer/blob/main/README.md",
        },
        "kaggle_handle": "kaggle://keras/t5/keras/t5_small_multi/2",
    },
    "t5_1.1_small": {
        "metadata": {
            "description": (""),
            "params": 60511616,
            "official_name": "T5 1.1",
            "path": "t5",
            "model_card": "https://github.com/google-research/text-to-text-transfer-transformer/blob/main/README.md",
        },
        "kaggle_handle": "kaggle://kerashub/t5-1.1/keras/t5_1.1_small",
    },
    "t5_base_multi": {
        "metadata": {
            "description": (
                "12-layer T5 model. Trained on the Colossal Clean Crawled "
                "Corpus (C4)."
            ),
            "params": 0,
            "official_name": "T5",
            "path": "t5",
            "model_card": "https://github.com/google-research/text-to-text-transfer-transformer/blob/main/README.md",
        },
        "kaggle_handle": "kaggle://keras/t5/keras/t5_base_multi/2",
    },
    "t5_1.1_base": {
        "metadata": {
            "description": (""),
            "params": 247577856,
            "official_name": "T5 1.1",
            "path": "t5",
            "model_card": "https://github.com/google-research/text-to-text-transfer-transformer/blob/main/README.md",
        },
        "kaggle_handle": "kaggle://kerashub/t5-1.1/keras/t5_1.1_base",
    },
    "t5_large_multi": {
        "metadata": {
            "description": (
                "24-layer T5 model. Trained on the Colossal Clean Crawled "
                "Corpus (C4)."
            ),
            "params": 0,
            "official_name": "T5",
            "path": "t5",
            "model_card": "https://github.com/google-research/text-to-text-transfer-transformer/blob/main/README.md",
        },
        "kaggle_handle": "kaggle://keras/t5/keras/t5_large_multi/2",
    },
    "t5_1.1_large": {
        "metadata": {
            "description": (""),
            "params": 750251008,
            "official_name": "T5 1.1",
            "path": "t5",
            "model_card": "https://github.com/google-research/text-to-text-transfer-transformer/blob/main/README.md",
        },
        "kaggle_handle": "kaggle://kerashub/t5-1.1/keras/t5_1.1_large",
    },
    "t5_1.1_xl": {
        "metadata": {
            "description": (""),
            "params": 2783959040,
            "official_name": "T5 1.1",
            "path": "t5",
            "model_card": "https://github.com/google-research/text-to-text-transfer-transformer/blob/main/README.md",
        },
        "kaggle_handle": "",
    },
    "flan_small_multi": {
        "metadata": {
            "description": (
                "8-layer T5 model. Trained on the Colossal Clean Crawled "
                "Corpus (C4)."
            ),
            "params": 0,
            "official_name": "T5",
            "path": "t5",
            "model_card": "https://github.com/google-research/text-to-text-transfer-transformer/blob/main/README.md",
        },
        "kaggle_handle": "kaggle://keras/t5/keras/flan_small_multi/2",
    },
    "flan_base_multi": {
        "metadata": {
            "description": (
                "12-layer T5 model. Trained on the Colossal Clean Crawled "
                "Corpus (C4)."
            ),
            "params": 0,
            "official_name": "T5",
            "path": "t5",
            "model_card": "https://github.com/google-research/text-to-text-transfer-transformer/blob/main/README.md",
        },
        "kaggle_handle": "kaggle://keras/t5/keras/flan_base_multi/2",
    },
    "flan_large_multi": {
        "metadata": {
            "description": (
                "24-layer T5 model. Trained on the Colossal Clean Crawled "
                "Corpus (C4)."
            ),
            "params": 0,
            "official_name": "T5",
            "path": "t5",
            "model_card": "https://github.com/google-research/text-to-text-transfer-transformer/blob/main/README.md",
        },
        "kaggle_handle": "kaggle://keras/t5/keras/flan_large_multi/2",
    },
}
