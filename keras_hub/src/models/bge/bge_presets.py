"""BGE model preset configurations."""

backbone_presets = {
    "bge_small_en_v1.5": {
        "metadata": {
            "description": (
                "12-layer BGE model where all input is lowercased. "
                "Fine-tuned for English sentence embeddings and retrieval. "
                "Trained by BAAI."
            ),
            "params": 33391104,
            "path": "bge",
        },
        "kaggle_handle": "kaggle://keras/bge/keras/bge_small_en_v1.5/1",
    },
    "bge_base_en_v1.5": {
        "metadata": {
            "description": (
                "12-layer BGE model where all input is lowercased. "
                "Fine-tuned for English sentence embeddings and retrieval. "
                "Trained by BAAI."
            ),
            "params": 109482240,
            "path": "bge",
        },
        "kaggle_handle": "kaggle://keras/bge/keras/bge_base_en_v1.5/1",
    },
    "bge_large_en_v1.5": {
        "metadata": {
            "description": (
                "24-layer BGE model where all input is lowercased. "
                "Fine-tuned for English sentence embeddings and retrieval. "
                "Trained by BAAI."
            ),
            "params": 335141888,
            "path": "bge",
        },
        "kaggle_handle": "kaggle://keras/bge/keras/bge_large_en_v1.5/1",
    },
}
