"""HRM-Text preset configurations."""

backbone_presets = {
    "hrm_text_1b": {
        "metadata": {
            "description": (
                "1B-parameter HRM-Text hierarchical recurrent model."
            ),
            "params": 1000000000,
            "path": "hrm_text",
        },
        "config": {
            "vocabulary_size": 65536,
            "hidden_dim": 1536,
            "intermediate_dim": 4096,
            "num_layers_per_stack": 16,
            "num_attention_heads": 12,
            "head_dim": 128,
            "h_cycles": 2,
            "l_cycles": 3,
            "max_sequence_length": 4096,
            "rope_theta": 10000.0,
            "rms_norm_epsilon": 1e-6,
            "embedding_scale": 39.191835884530846,
            "tie_word_embeddings": False,
        },
    }
}
