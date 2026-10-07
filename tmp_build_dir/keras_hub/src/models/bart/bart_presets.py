"""BART model preset configurations."""

backbone_presets = {
    "bart_base_en": {
        "metadata": {
            "description": (
                "6-layer BART model where case is maintained. "
                "Trained on BookCorpus, English Wikipedia and CommonCrawl."
            ),
            "params": 139417344,
            "path": "bart",
        },
        "kaggle_handle": "kaggle://keras/bart/keras/bart_base_en/3",
    },
    "bart_large_en": {
        "metadata": {
            "description": (
                "12-layer BART model where case is maintained. "
                "Trained on BookCorpus, English Wikipedia and CommonCrawl."
            ),
            "params": 406287360,
            "path": "bart",
        },
        "config": {
            "vocabulary_size": 50265,
            "num_layers": 12,
            "num_heads": 16,
            "hidden_dim": 1024,
            "intermediate_dim": 4096,
            "dropout": 0.1,
            "max_sequence_length": 1024,
        },
        "kaggle_handle": "kaggle://keras/bart/keras/bart_large_en/3",
    },
    "bart_large_en_cnn": {
        "metadata": {
            "description": (
                "The `bart_large_en` backbone model fine-tuned on the CNN+DM "
                "summarization dataset."
            ),
            "params": 406287360,
            "path": "bart",
        },
        "config": {
            "vocabulary_size": 50264,
            "num_layers": 12,
            "num_heads": 16,
            "hidden_dim": 1024,
            "intermediate_dim": 4096,
            "dropout": 0.1,
            "max_sequence_length": 1024,
        },
        "kaggle_handle": "kaggle://keras/bart/keras/bart_large_en_cnn/3",
    },
}
