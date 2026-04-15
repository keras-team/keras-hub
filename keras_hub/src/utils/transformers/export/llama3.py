import keras.ops as ops


def get_llama3_config(backbone):
    """Convert Keras Llama3 backbone config to Hugging Face config dictionary."""
    head_dim = backbone.hidden_dim // backbone.num_query_heads

    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": backbone.vocabulary_size,
        "num_hidden_layers": backbone.num_layers,
        "num_attention_heads": backbone.num_query_heads,
        "num_key_value_heads": backbone.num_key_value_heads,
        "hidden_size": backbone.hidden_dim,
        "intermediate_size": backbone.intermediate_dim,
        "head_dim": head_dim,
        "rms_norm_eps": backbone.layer_norm_epsilon,
        "rope_theta": backbone.rope_max_wavelength,
        "hidden_act": "silu",
        "attention_dropout": backbone.dropout,
        "attention_bias": False,
        "mlp_bias": False,
        "tie_word_embeddings": backbone.tie_word_embeddings,
        "use_cache": True,
        "torch_dtype": backbone.dtype_policy.name,
    }

    # Llama 3.1+ uses scaled RoPE ("llama3" rope_type).
    # Reconstruct the rope_scaling dict when the adjustment factor is set.
    if backbone.rope_frequency_adjustment_factor is not None:
        hf_config["rope_scaling"] = {
            "rope_type": "llama3",
            "factor": backbone.rope_frequency_adjustment_factor,
            "low_freq_factor": backbone.rope_low_freq_factor,
            "high_freq_factor": backbone.rope_high_freq_factor,
            "original_max_position_embeddings": (
                backbone.rope_pretraining_sequence_length
            ),
        }

    return hf_config


def get_llama3_weights_map(backbone, include_lm_head=False):
    """Create a Keras-to-HuggingFace weight name mapping for Llama3.

    Args:
        backbone: A `keras_hub.models.Llama3Backbone` instance.
        include_lm_head: If True, includes the ``lm_head.weight`` tensor
            (used when exporting a CausalLM task).

    Returns:
        dict mapping HuggingFace weight keys to Keras tensors.
    """
    weights_map = {}

    # Token embeddings
    token_embedding_layer = backbone.get_layer("token_embedding")
    weights_map["model.embed_tokens.weight"] = token_embedding_layer.embeddings

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"transformer_layer_{i}")
        attn_layer = decoder_layer._self_attention_layer

        # --- Normalization ---
        # Pre-attention (input) layernorm
        weights_map[f"model.layers.{i}.input_layernorm.weight"] = (
            decoder_layer._self_attention_layernorm.scale
        )
        # Pre-MLP (post-attention) layernorm
        weights_map[f"model.layers.{i}.post_attention_layernorm.weight"] = (
            decoder_layer._feedforward_layernorm.scale
        )

        # --- Attention projections ---
        # Keras Q kernel: (hidden_dim, num_query_heads, head_dim)
        # HF   q_proj.weight: (num_query_heads * head_dim, hidden_dim)
        q_kernel = attn_layer._query_dense.kernel
        q_kernel = ops.reshape(q_kernel, (backbone.hidden_dim, -1))
        weights_map[f"model.layers.{i}.self_attn.q_proj.weight"] = (
            ops.transpose(q_kernel)
        )

        # Keras K kernel: (hidden_dim, num_key_value_heads, head_dim)
        # HF   k_proj.weight: (num_key_value_heads * head_dim, hidden_dim)
        k_kernel = attn_layer._key_dense.kernel
        k_kernel = ops.reshape(k_kernel, (backbone.hidden_dim, -1))
        weights_map[f"model.layers.{i}.self_attn.k_proj.weight"] = (
            ops.transpose(k_kernel)
        )

        # Keras V kernel: (hidden_dim, num_key_value_heads, head_dim)
        # HF   v_proj.weight: (num_key_value_heads * head_dim, hidden_dim)
        v_kernel = attn_layer._value_dense.kernel
        v_kernel = ops.reshape(v_kernel, (backbone.hidden_dim, -1))
        weights_map[f"model.layers.{i}.self_attn.v_proj.weight"] = (
            ops.transpose(v_kernel)
        )

        # Keras O kernel: (num_query_heads, head_dim, hidden_dim)
        # HF   o_proj.weight: (hidden_dim, num_query_heads * head_dim)
        o_kernel = attn_layer._output_dense.kernel
        o_kernel = ops.reshape(o_kernel, (-1, backbone.hidden_dim))
        weights_map[f"model.layers.{i}.self_attn.o_proj.weight"] = (
            ops.transpose(o_kernel)
        )

        # --- MLP (SwiGLU) ---
        # Keras gate/up kernel: (hidden_dim, intermediate_dim)
        # HF   gate/up_proj.weight: (intermediate_dim, hidden_dim)
        gate_kernel = decoder_layer._feedforward_gate_dense.kernel
        weights_map[f"model.layers.{i}.mlp.gate_proj.weight"] = ops.transpose(
            gate_kernel
        )

        up_kernel = decoder_layer._feedforward_intermediate_dense.kernel
        weights_map[f"model.layers.{i}.mlp.up_proj.weight"] = ops.transpose(
            up_kernel
        )

        # Keras down kernel: (intermediate_dim, hidden_dim)
        # HF   down_proj.weight: (hidden_dim, intermediate_dim)
        down_kernel = decoder_layer._feedforward_output_dense.kernel
        weights_map[f"model.layers.{i}.mlp.down_proj.weight"] = ops.transpose(
            down_kernel
        )

    # Final layernorm
    weights_map["model.norm.weight"] = backbone.get_layer(
        "sequence_output_layernorm"
    ).scale

    # LM head (only when exporting the full CausalLM task)
    if include_lm_head:
        if backbone.tie_word_embeddings:
            weights_map["lm_head.weight"] = weights_map[
                "model.embed_tokens.weight"
            ]
        else:
            weights_map["lm_head.weight"] = ops.transpose(
                token_embedding_layer.reverse_embeddings
            )

    return weights_map


def get_llama3_tokenizer_config(tokenizer):
    """Build a HuggingFace-compatible tokenizer_config.json for Llama3."""
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": tokenizer.start_token,
        "eos_token": tokenizer.end_token,
        "pad_token": None,
        "unk_token": None,
        "add_bos_token": True,
        "add_eos_token": False,
        "clean_up_tokenization_spaces": True,
        "model_max_length": 131072,
    }
    return tokenizer_config
