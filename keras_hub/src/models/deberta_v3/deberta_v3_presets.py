"""DeBERTa model preset configurations."""

backbone_presets = {
    "deberta_v3_extra_small_en": {
        "metadata": {
            "description": (
                "12-layer DeBERTaV3 model where case is maintained. "
                "Trained on English Wikipedia, BookCorpus and OpenWebText."
            ),
            "params": 70682112,
            "official_name": "DeBERTaV3",
            "path": "deberta_v3",
            "model_card": "https://huggingface.co/microsoft/deberta-v3-xsmall",
        },
        "kaggle_handle": "kaggle://keras/deberta_v3/keras/deberta_v3_extra_small_en/2",
    },
    "deberta_v3_small_en": {
        "metadata": {
            "description": (
                "6-layer DeBERTaV3 model where case is maintained. "
                "Trained on English Wikipedia, BookCorpus and OpenWebText."
            ),
            "params": 141304320,
            "official_name": "DeBERTaV3",
            "path": "deberta_v3",
            "model_card": "https://huggingface.co/microsoft/deberta-v3-small",
        },
        "kaggle_handle": "kaggle://keras/deberta_v3/keras/deberta_v3_small_en/2",
    },
    "deberta_v3_base_en": {
        "metadata": {
            "description": (
                "12-layer DeBERTaV3 model where case is maintained. "
                "Trained on English Wikipedia, BookCorpus and OpenWebText."
            ),
            "params": 183831552,
            "official_name": "DeBERTaV3",
            "path": "deberta_v3",
            "model_card": "https://huggingface.co/microsoft/deberta-v3-base",
        },
        "kaggle_handle": "kaggle://keras/deberta_v3/keras/deberta_v3_base_en/2",
    },
    "deberta_v3_large_en": {
        "metadata": {
            "description": (
                "24-layer DeBERTaV3 model where case is maintained. "
                "Trained on English Wikipedia, BookCorpus and OpenWebText."
            ),
            "params": 434012160,
            "official_name": "DeBERTaV3",
            "path": "deberta_v3",
            "model_card": "https://huggingface.co/microsoft/deberta-v3-large",
        },
        "kaggle_handle": "kaggle://keras/deberta_v3/keras/deberta_v3_large_en/2",
    },
    "deberta_v3_base_multi": {
        "metadata": {
            "description": (
                "12-layer DeBERTaV3 model where case is maintained. "
                "Trained on the 2.5TB multilingual CC100 dataset."
            ),
            "params": 278218752,
            "official_name": "DeBERTaV3",
            "path": "deberta_v3",
            "model_card": "https://huggingface.co/microsoft/mdeberta-v3-base",
        },
        "kaggle_handle": "kaggle://keras/deberta_v3/keras/deberta_v3_base_multi/2",
    },
}
