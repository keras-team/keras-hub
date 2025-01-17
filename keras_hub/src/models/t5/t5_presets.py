"""T5 model preset configurations."""

backbone_presets = {
    "t5_small_multi": {
        "metadata": {
            "description": (
                "8-layer T5 model. Trained on the Colossal Clean Crawled "
                "Corpus (C4)."
            ),
            "params": 0,
            "path": "t5",
        },
        "kaggle_handle": "kaggle://keras/t5/keras/t5_small_multi/3",
    },
    "t5_1.1_small": {
        "metadata": {
            "description": (""),
            "params": 60511616,
            "path": "t5",
        },
        "kaggle_handle": "kaggle://keras/t5/keras/t5_1.1_small/2",
    },
    "t5_base_multi": {
        "metadata": {
            "description": (
                "12-layer T5 model. Trained on the Colossal Clean Crawled "
                "Corpus (C4)."
            ),
            "params": 0,
            "path": "t5",
        },
        "kaggle_handle": "kaggle://keras/t5/keras/t5_base_multi/3",
    },
    "t5_1.1_base": {
        "metadata": {
            "description": (""),
            "params": 247577856,
            "path": "t5",
        },
        "kaggle_handle": "kaggle://keras/t5/keras/t5_1.1_base/2",
    },
    "t5_large_multi": {
        "metadata": {
            "description": (
                "24-layer T5 model. Trained on the Colossal Clean Crawled "
                "Corpus (C4)."
            ),
            "params": 0,
            "path": "t5",
        },
        "kaggle_handle": "kaggle://keras/t5/keras/t5_large_multi/3",
    },
    "t5_1.1_large": {
        "metadata": {
            "description": (""),
            "params": 750251008,
            "path": "t5",
        },
        "kaggle_handle": "kaggle://keras/t5/keras/t5_1.1_large/2",
    },
    "t5_1.1_xl": {
        "metadata": {
            "description": (""),
            "params": 2849757184,
            "path": "t5",
        },
        "kaggle_handle": "kaggle://keras/t5/keras/t5_1.1_xl/2",
    },
    "t5_1.1_xxl": {
        "metadata": {
            "description": (""),
            "params": 11135332352,
            "path": "t5",
        },
        "kaggle_handle": "kaggle://keras/t5/keras/t5_1.1_xxl/2",
    },
    "flan_small_multi": {
        "metadata": {
            "description": (
                "8-layer T5 model. Trained on the Colossal Clean Crawled "
                "Corpus (C4)."
            ),
            "params": 0,
            "path": "t5",
        },
        "kaggle_handle": "kaggle://keras/t5/keras/flan_small_multi/3",
    },
    "flan_base_multi": {
        "metadata": {
            "description": (
                "12-layer T5 model. Trained on the Colossal Clean Crawled "
                "Corpus (C4)."
            ),
            "params": 0,
            "path": "t5",
        },
        "kaggle_handle": "kaggle://keras/t5/keras/flan_base_multi/3",
    },
    "flan_large_multi": {
        "metadata": {
            "description": (
                "24-layer T5 model. Trained on the Colossal Clean Crawled "
                "Corpus (C4)."
            ),
            "params": 0,
            "path": "t5",
        },
        "kaggle_handle": "kaggle://keras/t5/keras/flan_large_multi/3",
    },
}
