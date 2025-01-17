"""ALBERT model preset configurations."""

backbone_presets = {
    "albert_base_en_uncased": {
        "metadata": {
            "description": (
                "12-layer ALBERT model where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 11683584,
            "path": "albert",
        },
        "kaggle_handle": "kaggle://keras/albert/keras/albert_base_en_uncased/5",
    },
    "albert_large_en_uncased": {
        "metadata": {
            "description": (
                "24-layer ALBERT model where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 17683968,
            "path": "albert",
        },
        "kaggle_handle": "kaggle://keras/albert/keras/albert_large_en_uncased/3",
    },
    "albert_extra_large_en_uncased": {
        "metadata": {
            "description": (
                "24-layer ALBERT model where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 58724864,
            "path": "albert",
        },
        "kaggle_handle": "kaggle://keras/albert/keras/albert_extra_large_en_uncased/3",
    },
    "albert_extra_extra_large_en_uncased": {
        "metadata": {
            "description": (
                "12-layer ALBERT model where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 222595584,
            "path": "albert",
        },
        "kaggle_handle": "kaggle://keras/albert/keras/albert_extra_extra_large_en_uncased/3",
    },
}
