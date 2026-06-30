"""XLM-RoBERTa model preset configurations."""

backbone_presets = {
    "xlm_roberta_base_multi": {
        "metadata": {
            "description": (
                "12-layer XLM-RoBERTa model where case is maintained. "
                "Trained on CommonCrawl in 100 languages."
            ),
            "params": 277450752,
            "path": "xlm_roberta",
        },
        "kaggle_handle": "kaggle://keras/xlm_roberta/keras/xlm_roberta_base_multi/3",
    },
    "xlm_roberta_large_multi": {
        "metadata": {
            "description": (
                "24-layer XLM-RoBERTa model where case is maintained. "
                "Trained on CommonCrawl in 100 languages."
            ),
            "params": 558837760,
            "path": "xlm_roberta",
        },
        "kaggle_handle": "kaggle://keras/xlm_roberta/keras/xlm_roberta_large_multi/3",
    },
    "bge_m3": {
        "metadata": {
            "description": (
                "568M-parameter multilingual text embedding model supporting "
                "100+ languages with sequences up to 8192 tokens. Uses CLS "
                "token pooling with L2 normalization. Supports dense, sparse, "
                "and multi-vector (ColBERT-style) retrieval. From BAAI."
            ),
            "params": 566702080,
            "path": "xlm_roberta",
        },
        "kaggle_handle": "kaggle://keras/bge/keras/bge_m3/1",
    },
    "multilingual_e5_base": {
        "metadata": {
            "description": (
                "12-layer multilingual E5 embedding model with 768-dimensional "
                "vectors. Fine-tuned for dense retrieval across 100+ languages "
                "using weakly-supervised contrastive pre-training. "
                "Prefix inputs with 'query: ' for queries and 'passage: ' "
                "for documents."
            ),
            "params": 277450752,
            "path": "xlm_roberta",
        },
        "kaggle_handle": "kaggle://keras/multilingual-e5/keras/multilingual_e5_base/1",
    },
    "multilingual_e5_large": {
        "metadata": {
            "description": (
                "24-layer multilingual E5 embedding model with 1024-dimensional"
                "vectors. Fine-tuned for dense retrieval across 100+ languages "
                "using weakly-supervised contrastive pre-training. "
                "Prefix inputs with 'query: ' for queries and 'passage: ' "
                "for documents."
            ),
            "params": 558837760,
            "path": "xlm_roberta",
        },
        "kaggle_handle": "kaggle://keras/multilingual-e5/keras/multilingual_e5_large/1",
    },
}
