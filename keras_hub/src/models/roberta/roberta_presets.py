"""RoBERTa model preset configurations."""

backbone_presets = {
    "roberta_base_en": {
        "metadata": {
            "description": (
                "12-layer RoBERTa model where case is maintained."
                "Trained on English Wikipedia, BooksCorpus, CommonCraw, and "
                "OpenWebText."
            ),
            "params": 124052736,
            "path": "roberta",
        },
        "kaggle_handle": "kaggle://keras/roberta/keras/roberta_base_en/3",
    },
    "roberta_large_en": {
        "metadata": {
            "description": (
                "24-layer RoBERTa model where case is maintained."
                "Trained on English Wikipedia, BooksCorpus, CommonCraw, and "
                "OpenWebText."
            ),
            "params": 354307072,
            "path": "roberta",
        },
        "kaggle_handle": "kaggle://keras/roberta/keras/roberta_large_en/3",
    },
}
