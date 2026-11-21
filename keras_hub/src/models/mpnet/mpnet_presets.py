"""MPNet model preset configurations."""

backbone_presets = {
    # Sentence-transformer models (MPNet backbone, fine-tuned for embeddings).
    "all_mpnet_base_v2": {
        "metadata": {
            "description": (
                "12-layer MPNet sentence embedding model. Maps sentences and "
                "paragraphs to 768-dimensional normalized vectors. Trained on "
                "1B+ sentence pairs for semantic search, clustering, and "
                "similarity."
            ),
            "params": 108894336,
            "path": "mpnet",
        },
        "kaggle_handle": "kaggle://keras/mpnet/keras/all_mpnet_base_v2/1",
    },
    "nli_mpnet_base_v2": {
        "metadata": {
            "description": (
                "12-layer MPNet sentence embedding model trained on natural "
                "language inference (NLI) data. Maps sentences to "
                "768-dimensional dense vectors that are not normalized. "
                "Inputs are truncated to 75 tokens."
            ),
            "params": 108894336,
            "path": "mpnet",
        },
        "kaggle_handle": "kaggle://keras/mpnet/keras/nli_mpnet_base_v2/1",
    },
    "paraphrase_mpnet_base_v2": {
        "metadata": {
            "description": (
                "12-layer MPNet sentence embedding model trained on "
                "paraphrase data. Maps sentences and paragraphs to "
                "768-dimensional dense vectors that are not normalized, for "
                "paraphrase mining, clustering, and semantic search."
            ),
            "params": 108894336,
            "path": "mpnet",
        },
        "kaggle_handle": "kaggle://keras/mpnet/keras/paraphrase_mpnet_base_v2/1",
    },
}
