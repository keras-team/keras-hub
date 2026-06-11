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
    # BGE family: BAAI General Embedding models optimized for dense retrieval.
    # Use pooling_mode="cls" with L2 normalization.
    "bge_small_en_v1.5": {
        "metadata": {
            "description": (
                "12-layer BGE small English embedding model (v1.5). Maps "
                "sentences to 384-dimensional L2-normalized dense vectors. "
                "Optimized for dense retrieval and semantic similarity."
            ),
            "params": 33360000,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bge/keras/bge_small_en_v1.5/1",
    },
    "bge_base_en_v1.5": {
        "metadata": {
            "description": (
                "12-layer BGE base English embedding model (v1.5). Maps "
                "sentences to 768-dimensional L2-normalized dense vectors. "
                "Optimized for dense retrieval and semantic similarity."
            ),
            "params": 109482240,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bge/keras/bge_base_en_v1.5/1",
    },
    "bge_large_en_v1.5": {
        "metadata": {
            "description": (
                "24-layer BGE large English embedding model (v1.5). Maps "
                "sentences to 1024-dimensional L2-normalized dense vectors. "
                "Highest accuracy in the BGE English family."
            ),
            "params": 335141888,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bge/keras/bge_large_en_v1.5/1",
    },
}
