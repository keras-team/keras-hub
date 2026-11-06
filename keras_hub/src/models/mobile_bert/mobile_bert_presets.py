"""MobileBERT model preset configurations."""

backbone_presets = {
    "mobilebert_uncased": {
        "metadata": {
            "description": (
                "24-layer MobileBERT model where all text is lowercased. "
                "Trained on English Wikipedia and BooksCorpus."
            ),
            "params": 24581888,
            "path": "mobilebert",
        },
        "kaggle_handle": "kaggle://keras/mobilebert/keras/mobilebert_uncased/1",
    },
}
