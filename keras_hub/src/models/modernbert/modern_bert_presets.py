"""ModernBERT model preset configurations."""

backbone_presets = {
    "modernbert_base_en": {
        "metadata": {
            "description": (
                "22-layer ModernBERT Base encoder model pretrained on "
                "English for masked language modeling. Uses Rotary Position "
                "Embeddings (RoPE), alternating local and global attention, "
                "and GeGLU feedforward layers."
            ),
            "params": 149014272,
            "path": "modernbert",
        },
        "kaggle_handle": (
            "kaggle://keras/modernbert/keras/modernbert_base_en/1"
        ),
    },
    "modernbert_large_en": {
        "metadata": {
            "description": (
                "28-layer ModernBERT Large encoder model pretrained on "
                "English for masked language modeling. Uses Rotary Position "
                "Embeddings (RoPE), alternating local and global attention, "
                "and GeGLU feedforward layers."
            ),
            "params": 394781696,
            "path": "modernbert",
        },
        "kaggle_handle": (
            "kaggle://keras/modernbert/keras/modernbert_large_en/1"
        ),
    },
}
