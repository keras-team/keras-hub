"""ELECTRA model preset configurations."""

backbone_presets = {
    "electra_small_discriminator_uncased_en": {
        "metadata": {
            "description": (
                "12-layer small ELECTRA discriminator model. All inputs are "
                "lowercased. Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 13548800,
            "path": "electra",
        },
        "kaggle_handle": "kaggle://keras/electra/keras/electra_small_discriminator_uncased_en/2",
    },
    "electra_small_generator_uncased_en": {
        "metadata": {
            "description": (
                "12-layer small ELECTRA generator model. All inputs are "
                "lowercased. Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 13548800,
            "path": "electra",
        },
        "kaggle_handle": "kaggle://keras/electra/keras/electra_small_generator_uncased_en/2",
    },
    "electra_base_discriminator_uncased_en": {
        "metadata": {
            "description": (
                "12-layer base ELECTRA discriminator model. All inputs are "
                "lowercased. Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 109482240,
            "path": "electra",
        },
        "kaggle_handle": "kaggle://keras/electra/keras/electra_base_discriminator_uncased_en/2",
    },
    "electra_base_generator_uncased_en": {
        "metadata": {
            "description": (
                "12-layer base ELECTRA generator model. All inputs are "
                "lowercased. Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 33576960,
            "path": "electra",
        },
        "kaggle_handle": "kaggle://keras/electra/keras/electra_base_generator_uncased_en/2",
    },
    "electra_large_discriminator_uncased_en": {
        "metadata": {
            "description": (
                "24-layer large ELECTRA discriminator model. All inputs are "
                "lowercased. Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 335141888,
            "path": "electra",
        },
        "kaggle_handle": "kaggle://keras/electra/keras/electra_large_discriminator_uncased_en/2",
    },
    "electra_large_generator_uncased_en": {
        "metadata": {
            "description": (
                "24-layer large ELECTRA generator model. All inputs are "
                "lowercased. Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 51065344,
            "path": "electra",
        },
        "kaggle_handle": "kaggle://keras/electra/keras/electra_large_generator_uncased_en/2",
    },
}
