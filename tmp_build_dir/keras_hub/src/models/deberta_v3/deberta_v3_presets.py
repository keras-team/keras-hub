"""DeBERTa model preset configurations."""

backbone_presets = {
    "deberta_v3_extra_small_en": {
        "metadata": {
            "description": (
                "12-layer DeBERTaV3 model where case is maintained. "
                "Trained on English Wikipedia, BookCorpus and OpenWebText."
            ),
            "params": 70682112,
            "path": "deberta_v3",
        },
        "kaggle_handle": "kaggle://keras/deberta_v3/keras/deberta_v3_extra_small_en/3",
    },
    "deberta_v3_small_en": {
        "metadata": {
            "description": (
                "6-layer DeBERTaV3 model where case is maintained. "
                "Trained on English Wikipedia, BookCorpus and OpenWebText."
            ),
            "params": 141304320,
            "path": "deberta_v3",
        },
        "kaggle_handle": "kaggle://keras/deberta_v3/keras/deberta_v3_small_en/3",
    },
    "deberta_v3_base_en": {
        "metadata": {
            "description": (
                "12-layer DeBERTaV3 model where case is maintained. "
                "Trained on English Wikipedia, BookCorpus and OpenWebText."
            ),
            "params": 183831552,
            "path": "deberta_v3",
        },
        "kaggle_handle": "kaggle://keras/deberta_v3/keras/deberta_v3_base_en/3",
    },
    "deberta_v3_large_en": {
        "metadata": {
            "description": (
                "24-layer DeBERTaV3 model where case is maintained. "
                "Trained on English Wikipedia, BookCorpus and OpenWebText."
            ),
            "params": 434012160,
            "path": "deberta_v3",
        },
        "kaggle_handle": "kaggle://keras/deberta_v3/keras/deberta_v3_large_en/3",
    },
    "deberta_v3_base_multi": {
        "metadata": {
            "description": (
                "12-layer DeBERTaV3 model where case is maintained. "
                "Trained on the 2.5TB multilingual CC100 dataset."
            ),
            "params": 278218752,
            "path": "deberta_v3",
        },
        "kaggle_handle": "kaggle://keras/deberta_v3/keras/deberta_v3_base_multi/3",
    },
}
