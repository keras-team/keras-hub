"""ModernBERT model preset configurations."""

backbone_presets = {
    "modern_bert_tiny": {
        "metadata": {
            "description": "6-layer ModernBERT model for testing.",
            "params": 22400000,
            "official_name": "ModernBERT",
            "path": "modernbert",
        },
        "config": {
            "vocabulary_size": 50368,
            "hidden_dim": 256,
            "intermediate_dim": 384, # 1.5x hidden_dim
            "num_layers": 6,
            "num_heads": 8,
            "dropout": 0.1,
            "local_attention_window": 128,
            "global_attn_every_n_layers": 3,
            "rotary_max_wavelength": 160000,
            "layer_norm_epsilon": 1e-5,
        },
        "preprocessor_config": {
            "sequence_length": 512,
        },
        "weights_url": None, 
        "weights_hash": None,
    },
    "modern_bert_base": {
        "metadata": {
            "description": (
                "ModernBERT base model with 22 layers and 768 hidden dimension. "
                "Optimized for speed and accuracy with an 8k context window."
            ),
            "params": 149000000,
            "official_name": "ModernBERT",
            "path": "modernbert",
        },
        "config": {
            "vocabulary_size": 50368,
            "hidden_dim": 768,
            "intermediate_dim": 1152, # 1.5x hidden_dim
            "num_layers": 22,
            "num_heads": 12,
            "dropout": 0.0,
            "local_attention_window": 128,
            "global_attn_every_n_layers": 3,
            "rotary_max_wavelength": 160000,
            "layer_norm_epsilon": 1e-5,
        },
        "preprocessor_config": {
            "sequence_length": 8192,
        },
        "weights_url": None,
        "weights_hash": None,
    },
    "modern_bert_large": {
        "metadata": {
            "description": (
                "ModernBERT large model with 28 layers and 1024 hidden dimension. "
                "State-of-the-art performance for large-scale encoder tasks."
            ),
            "params": 395000000,
            "official_name": "ModernBERT",
            "path": "modernbert",
        },
        "config": {
            "vocabulary_size": 50368,
            "hidden_dim": 1024,
            "intermediate_dim": 2048, # Note: Large uses 2x hidden_dim
            "num_layers": 28,
            "num_heads": 16,
            "dropout": 0.0,
            "local_attention_window": 128,
            "global_attn_every_n_layers": 3,
            "rotary_max_wavelength": 160000,
            "layer_norm_epsilon": 1e-5,
        },
        "preprocessor_config": {
            "sequence_length": 8192,
        },
        "weights_url": None,
        "weights_hash": None,
    },
}