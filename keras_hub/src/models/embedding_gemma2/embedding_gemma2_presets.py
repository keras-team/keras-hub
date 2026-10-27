"""EmbeddingGemma2 model preset configurations."""

backbone_presets = {
    "embedding_gemma2": {
        "metadata": {
            "description": (
                "EmbeddingGemma 2: 744M-parameter, 24-layer multimodal "
                "embedding model built on Gemma 4. Embeds text, images, "
                "video and audio into a 768-dimensional space."
            ),
            "params": 744371992,
            "path": "embedding_gemma2",
        },
        "kaggle_handle": "kaggle://keras/embedding-gemma2/keras/embedding_gemma2/1",
    }
}
