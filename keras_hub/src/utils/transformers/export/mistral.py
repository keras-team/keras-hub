import keras.ops as ops


def get_mistral_config(backbone):
    """Convert Keras Mistral backbone config to Hugging Face dictionary."""
    head_dim = backbone.hidden_dim // backbone.num_query_heads

    hf_config = {
        "architectures": ["MistralForCausalLM"],
        "model_type": "mistral",
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
        "tie_word_embeddings": False,
        "use_cache": True,
        "sliding_window": backbone.sliding_window,
        "torch_dtype": backbone.dtype_policy.name,
    }

    return hf_config


def get_mistral_weights_map(backbone, include_lm_head=False):
    """Create a Keras-to-HuggingFace weight name mapping for Mistral.

    Args:
        backbone: A `keras_hub.models.MistralBackbone` instance.
        include_lm_head: If True, includes the ``lm_head.weight`` tensor
            (used when exporting a CausalLM task).

    Returns:
        dict mapping HuggingFace weight keys to Keras tensors.
    """
    weights_map = {}

    # Token embeddings
    weights_map["model.embed_tokens.weight"] = (
        backbone.token_embedding.embeddings
    )

    for i in range(backbone.num_layers):
        decoder_layer = backbone.transformer_layers[i]
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
    weights_map["model.norm.weight"] = backbone.layer_norm.scale

    # LM head (only when exporting the full CausalLM task)
    if include_lm_head:
        # Mistral models typically don't tie embeddings
        weights_map["lm_head.weight"] = ops.transpose(
            backbone.token_embedding.reverse_embeddings
        )

    return weights_map


def get_mistral_tokenizer_config(tokenizer):
    """Build a HuggingFace-compatible tokenizer_config.json for Mistral."""
    tokenizer_config = {
        "add_bos_token": True,
        "add_eos_token": False,
        "added_tokens_decoder": {},
        "bos_token": tokenizer.start_token,
        "clean_up_tokenization_spaces": False,
        "eos_token": tokenizer.end_token,
        "legacy": False,
        "model_max_length": 32768,
        "pad_token": None,
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "LlamaTokenizer",
        "unk_token": "<unk>",
        "use_default_system_prompt": False,
    }

    # Add added_tokens_decoder
    added_tokens_decoder = {}
    special_tokens = [tokenizer.start_token, tokenizer.end_token, "<unk>"]
    for token in special_tokens:
        token_id = tokenizer.token_to_id(token)
        if token_id is not None:
            added_tokens_decoder[str(token_id)] = {
                "content": token,
                "special": True,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
            }
    tokenizer_config["added_tokens_decoder"] = added_tokens_decoder
    return tokenizer_config
