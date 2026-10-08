import keras.ops as ops


def get_qwen_config(backbone):
    """Convert Keras Qwen config to Hugging Face Qwen2Config."""
    return {
        # Core architectural dimensions
        "vocab_size": backbone.vocabulary_size,
        "hidden_size": backbone.hidden_dim,
        "num_hidden_layers": backbone.num_layers,
        "num_attention_heads": backbone.num_query_heads,
        "num_key_value_heads": backbone.num_key_value_heads,
        "intermediate_size": backbone.intermediate_dim,
        # Activation and regularization
        "hidden_act": "silu",
        "attention_dropout": backbone.dropout,
        # Numerical stability and initialization
        "rms_norm_eps": backbone.layer_norm_epsilon,
        "initializer_range": 0.02,
        # RoPE settings
        "rope_theta": backbone.rope_max_wavelength,
        # Model behavior
        "use_cache": True,
        "tie_word_embeddings": backbone.tie_word_embeddings,
        "model_type": "qwen2",
    }


def get_qwen_weights_map(backbone, include_lm_head=False):
    """Create a weights map for a given Qwen model."""
    weights_map = {}

    # 1. Embeddings
    weights_map["model.embed_tokens.weight"] = backbone.get_layer(
        "token_embedding"
    ).embeddings

    for i in range(backbone.num_layers):
        # Access the decoder layer
        decoder_layer = backbone.get_layer(f"transformer_layer_{i}")

        # --- Normalization ---
        # Input Norm (Pre-Attention)
        weights_map[f"model.layers.{i}.input_layernorm.weight"] = (
            decoder_layer._self_attention_layernorm.scale
        )

        # Post Attention Norm (Pre-MLP)
        weights_map[f"model.layers.{i}.post_attention_layernorm.weight"] = (
            decoder_layer._feedforward_layernorm.scale
        )

        # --- Attention ---
        attn_layer = decoder_layer._self_attention_layer

        # Query
        q_kernel = attn_layer._query_dense.kernel
        q_kernel = ops.reshape(q_kernel, (backbone.hidden_dim, -1))
        weights_map[f"model.layers.{i}.self_attn.q_proj.weight"] = (
            ops.transpose(q_kernel)
        )
        weights_map[f"model.layers.{i}.self_attn.q_proj.bias"] = ops.reshape(
            attn_layer._query_dense.bias, (-1,)
        )

        # Key
        k_kernel = attn_layer._key_dense.kernel
        k_kernel = ops.reshape(k_kernel, (backbone.hidden_dim, -1))
        weights_map[f"model.layers.{i}.self_attn.k_proj.weight"] = (
            ops.transpose(k_kernel)
        )
        weights_map[f"model.layers.{i}.self_attn.k_proj.bias"] = ops.reshape(
            attn_layer._key_dense.bias, (-1,)
        )

        # Value
        v_kernel = attn_layer._value_dense.kernel
        v_kernel = ops.reshape(v_kernel, (backbone.hidden_dim, -1))
        weights_map[f"model.layers.{i}.self_attn.v_proj.weight"] = (
            ops.transpose(v_kernel)
        )
        weights_map[f"model.layers.{i}.self_attn.v_proj.bias"] = ops.reshape(
            attn_layer._value_dense.bias, (-1,)
        )

        # Output
        o_kernel = attn_layer._output_dense.kernel
        o_kernel = ops.reshape(o_kernel, (-1, backbone.hidden_dim))
        weights_map[f"model.layers.{i}.self_attn.o_proj.weight"] = (
            ops.transpose(o_kernel)
        )

        # --- MLP (SwiGLU) ---
        gate_kernel = decoder_layer._feedforward_gate_dense.kernel
        weights_map[f"model.layers.{i}.mlp.gate_proj.weight"] = ops.transpose(
            gate_kernel
        )

        up_kernel = decoder_layer._feedforward_intermediate_dense.kernel
        weights_map[f"model.layers.{i}.mlp.up_proj.weight"] = ops.transpose(
            up_kernel
        )

        down_kernel = decoder_layer._feedforward_output_dense.kernel
        weights_map[f"model.layers.{i}.mlp.down_proj.weight"] = ops.transpose(
            down_kernel
        )

    # Final Norm
    weights_map["model.norm.weight"] = backbone.get_layer(
        "sequence_output_layernorm"
    ).scale

    # LM Head
    if include_lm_head:
        if backbone.tie_word_embeddings:
            weights_map["lm_head.weight"] = weights_map[
                "model.embed_tokens.weight"
            ]
        else:
            lm_head_w = backbone.get_layer("token_embedding").reverse_embeddings
            weights_map["lm_head.weight"] = ops.transpose(lm_head_w)

    return weights_map


def get_qwen_tokenizer_config(tokenizer):
    """Convert Keras Qwen tokenizer config to Hugging Face."""
    return {
        "tokenizer_class": "Qwen2Tokenizer",
        "bos_token": None,
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "unk_token": None,
        "model_max_length": 32768,
    }
