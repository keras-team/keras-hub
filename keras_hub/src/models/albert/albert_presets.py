"""ALBERT model preset configurations."""

backbone_presets = {
    "albert_base_en_uncased": {
        "metadata": {
            "description": (
                "12-layer ALBERT model where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 11683584,
            "official_name": "ALBERT",
            "path": "albert",
            "model_card": "https://github.com/google-research/albert/blob/master/README.md",
        },
        "kaggle_handle": "kaggle://keras/albert/keras/albert_base_en_uncased/2",
    },
    "albert_large_en_uncased": {
        "metadata": {
            "description": (
                "24-layer ALBERT model where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 17683968,
            "official_name": "ALBERT",
            "path": "albert",
            "model_card": "https://github.com/google-research/albert/blob/master/README.md",
        },
        "kaggle_handle": "kaggle://keras/albert/keras/albert_large_en_uncased/2",
    },
    "albert_extra_large_en_uncased": {
        "metadata": {
            "description": (
                "24-layer ALBERT model where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 58724864,
            "official_name": "ALBERT",
            "path": "albert",
            "model_card": "https://github.com/google-research/albert/blob/master/README.md",
        },
        "kaggle_handle": "kaggle://keras/albert/keras/albert_extra_large_en_uncased/2",
    },
    "albert_extra_extra_large_en_uncased": {
        "metadata": {
            "description": (
                "12-layer ALBERT model where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 222595584,
            "official_name": "ALBERT",
            "path": "albert",
            "model_card": "https://github.com/google-research/albert/blob/master/README.md",
        },
        "kaggle_handle": "kaggle://keras/albert/keras/albert_extra_extra_large_en_uncased/2",
    },
}
