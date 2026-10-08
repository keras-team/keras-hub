"""ModernBERT model preset configurations."""

backbone_presets = {
    "modern_bert_base_en": {
        "metadata": {
            "description": (
                "22-layer ModernBERT Base encoder model pretrained on "
                "English for masked language modeling. Uses Rotary Position "
                "Embeddings (RoPE), alternating local and global attention, "
                "and GeGLU feedforward layers."
            ),
            "params": 149014272,
            "path": "modern_bert",
        },
        "kaggle_handle": (
            "kaggle://keras/modernbert/keras/modern_bert_base_en/1"
        ),
    },
    "modern_bert_large_en": {
        "metadata": {
            "description": (
                "28-layer ModernBERT Large encoder model pretrained on "
                "English for masked language modeling. Uses Rotary Position "
                "Embeddings (RoPE), alternating local and global attention, "
                "and GeGLU feedforward layers."
            ),
            "params": 394781696,
            "path": "modern_bert",
        },
        "kaggle_handle": (
            "kaggle://keras/modernbert/keras/modern_bert_large_en/1"
        ),
    },
}
