"""Qwen3-ASR model preset configurations."""

backbone_presets = {
    "qwen3_asr_1.7b": {
        "metadata": {
            "description": (
                "Qwen3-ASR 1.7B model supporting 52 languages and dialects. "
                "Uses 1024 hidden size for the audio encoder."
            ),
            "params": 2000000000,
            "path": "qwen3_asr",
        },
        "config": {
            "vocabulary_size": 151936,
            "num_layers": 28,
            "num_query_heads": 16,
            "num_key_value_heads": 16,
            "head_dim": 64,
            "hidden_dim": 1024,
            "intermediate_dim": 2816,
            "audio_encoder": {
                "class_name": "keras_hub>Qwen3ASRAudioEncoder",
                "config": {
                    "d_model": 1024,
                    "encoder_layers": 24,
                    "encoder_attention_heads": 16,
                    "encoder_ffn_dim": 4096,
                    "downsample_hidden_size": 1024,
                    "num_mel_bins": 128,
                    "output_dim": 1024,
                },
            },
        },
        "kaggle_handle": "kaggle://keras/qwen3-asr/keras/qwen3_asr_1.7b/1",
    },
    "qwen3_asr_0.6b": {
        "metadata": {
            "description": (
                "Qwen3-ASR 0.6B model supporting 52 languages and dialects. "
                "Uses 896 hidden size for the audio encoder."
            ),
            "params": 900000000,
            "path": "qwen3_asr",
        },
        "config": {
            "vocabulary_size": 151936,
            "num_layers": 24,
            "num_query_heads": 14,
            "num_key_value_heads": 14,
            "head_dim": 64,
            "hidden_dim": 896,
            "intermediate_dim": 2432,
            "audio_encoder": {
                "class_name": "keras_hub>Qwen3ASRAudioEncoder",
                "config": {
                    "d_model": 896,
                    "encoder_layers": 20,
                    "encoder_attention_heads": 14,
                    "encoder_ffn_dim": 3584,
                    "downsample_hidden_size": 896,
                    "num_mel_bins": 128,
                    "output_dim": 896,
                },
            },
        },
        "kaggle_handle": "kaggle://keras/qwen3-asr/keras/qwen3_asr_0.6b/1",
    },
}
