"""BERT model preset configurations."""

backbone_presets = {
    "bert_tiny_en_uncased": {
        "metadata": {
            "description": (
                "2-layer BERT model where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 4385920,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bert/keras/bert_tiny_en_uncased/3",
    },
    "bert_small_en_uncased": {
        "metadata": {
            "description": (
                "4-layer BERT model where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 28763648,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bert/keras/bert_small_en_uncased/3",
    },
    "bert_medium_en_uncased": {
        "metadata": {
            "description": (
                "8-layer BERT model where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 41373184,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bert/keras/bert_medium_en_uncased/3",
    },
    "bert_base_en_uncased": {
        "metadata": {
            "description": (
                "12-layer BERT model where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 109482240,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bert/keras/bert_base_en_uncased/3",
    },
    "bert_base_en": {
        "metadata": {
            "description": (
                "12-layer BERT model where case is maintained. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 108310272,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bert/keras/bert_base_en/3",
    },
    "bert_base_zh": {
        "metadata": {
            "description": (
                "12-layer BERT model. Trained on Chinese Wikipedia."
            ),
            "params": 102267648,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bert/keras/bert_base_zh/3",
    },
    "bert_base_multi": {
        "metadata": {
            "description": (
                "12-layer BERT model where case is maintained. Trained on "
                "trained on Wikipedias of 104 languages"
            ),
            "params": 177853440,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bert/keras/bert_base_multi/3",
    },
    "bert_large_en_uncased": {
        "metadata": {
            "description": (
                "24-layer BERT model where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 335141888,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bert/keras/bert_large_en_uncased/3",
    },
    "bert_large_en": {
        "metadata": {
            "description": (
                "24-layer BERT model where case is maintained. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 333579264,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bert/keras/bert_large_en/3",
    },
    "bert_tiny_en_uncased_sst2": {
        "metadata": {
            "description": (
                "The bert_tiny_en_uncased backbone model fine-tuned on the "
                "SST-2 sentiment analysis dataset."
            ),
            "params": 4385920,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bert/keras/bert_tiny_en_uncased_sst2/5",
    },
}
