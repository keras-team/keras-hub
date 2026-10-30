"""MPNet model preset configurations."""

backbone_presets = {
    # Sentence-transformer models (MPNet backbone, fine-tuned for embeddings).
    "all_mpnet_base_v2_en": {
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
        "kaggle_handle": "kaggle://keras/sentence-transformers/keras/all_mpnet_base_v2_en/1",
    },
}
