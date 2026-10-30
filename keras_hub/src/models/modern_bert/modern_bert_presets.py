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
    "modern_bert_embed_base_en": {
        "metadata": {
            "description": (
                "22-layer ModernBERT Base sentence embedding model from Nomic "
                "AI (nomic-ai/modernbert-embed-base). Produces "
                "768-dimensional, L2-normalized, mean-pooled embeddings for "
                "inputs of up to 8192 tokens. Prefix queries with "
                "'search_query: ' and documents with 'search_document: '."
            ),
            "params": 149014272,
            "path": "modern_bert",
        },
        "kaggle_handle": (
            "kaggle://keras/modernbert-embed/keras/modern_bert_embed_base_en/1"
        ),
    },
    "gte_modern_bert_base_en": {
        "metadata": {
            "description": (
                "22-layer ModernBERT Base sentence embedding model from "
                "Alibaba-NLP (Alibaba-NLP/gte-modernbert-base). Produces "
                "768-dimensional [CLS]-pooled embeddings that are not "
                "normalized, so compare them with cosine similarity. The "
                "model supports inputs of up to 8192 tokens."
            ),
            "params": 149014272,
            "path": "modern_bert",
        },
        "kaggle_handle": (
            "kaggle://keras/gte-modernbert/keras/gte_modern_bert_base_en/1"
        ),
    },
}
