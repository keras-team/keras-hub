import keras.ops as ops


def get_qwen3_config(backbone):
    """Convert Keras Qwen3 config to Hugging Face Qwen2Config."""
    # Qwen3 uses the Qwen2 architecture (RoPE, SwiGLU, RMSNorm)
    cfg = backbone.get_config()

    return {
        # Core dimensions
        "vocab_size": cfg["vocabulary_size"],
        "hidden_size": cfg["hidden_dim"],
        "num_hidden_layers": cfg["num_layers"],
        "num_attention_heads": cfg["num_query_heads"],
        "num_key_value_heads": cfg["num_key_value_heads"],
        "intermediate_size": cfg["intermediate_dim"],
        
        # Architecture details
        "hidden_act": "silu",
        "rms_norm_eps": cfg["layer_norm_epsilon"],
        "rope_theta": cfg["rope_max_wavelength"],
        "tie_word_embeddings": cfg["tie_word_embeddings"],
        
        # Defaults
        "initializer_range": 0.02,
        "use_cache": True,
        "attention_dropout": cfg["dropout"],
        # HF uses "qwen2" model type for the Qwen family
        "model_type": "qwen2", 
    }


def get_qwen3_weights_map(backbone, include_lm_head=False):
    """Create a weights map for a given Qwen3 model."""
    weights_map = {}

    # 1. Embeddings
    weights_map["model.embed_tokens.weight"] = backbone.get_layer(
        "token_embedding"
    ).embeddings

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"transformer_layer_{i}")

        # KerasHub Qwen3LayerNorm uses 'scale'
        weights_map[f"model.layers.{i}.input_layernorm.weight"] = (
            decoder_layer._self_attention_layernorm.scale
        )

        weights_map[f"model.layers.{i}.post_attention_layernorm.weight"] = (
            decoder_layer._feedforward_layernorm.scale
        )

        # --- Attention ---
        attn_layer = decoder_layer._self_attention_layer

        # Helper to map QKV (Reshape -> Transpose -> Bias)
        def map_qkv(keras_layer, hf_name):
            # Kernel: (Hidden, Heads, Dim) -> (Hidden, Heads*Dim) 
            # -> (Heads*Dim, Hidden)
            k = ops.reshape(keras_layer.kernel, (backbone.hidden_dim, -1))
            weights_map[f"model.layers.{i}.self_attn.{hf_name}.weight"] = ops.transpose(k)
            
            # Bias: (Heads, Dim) -> (Heads*Dim)
            # Qwen usually includes biases for Q, K, V
            if keras_layer.bias is not None:
                b = ops.reshape(keras_layer.bias, (-1,))
                weights_map[f"model.layers.{i}.self_attn.{hf_name}.bias"] = b

        # Access sub-layers (Robust check for _underscore vs public)
        # Based on Qwen2, these are usually _query_dense, etc.
        q_layer = getattr(attn_layer, "query_dense", getattr(attn_layer, "_query_dense", None))
        k_layer = getattr(attn_layer, "key_dense", getattr(attn_layer, "_key_dense", None))
        v_layer = getattr(attn_layer, "value_dense", getattr(attn_layer, "_value_dense", None))
        o_layer = getattr(attn_layer, "output_dense", getattr(attn_layer, "_output_dense", None))

        if q_layer: map_qkv(q_layer, "q_proj")
        if k_layer: map_qkv(k_layer, "k_proj")
        if v_layer: map_qkv(v_layer, "v_proj")

        # Output (O_Proj) - Qwen usually has NO BIAS on output
        if o_layer:
            # Kernel: (Heads, Dim, Hidden) -> (Heads*Dim, Hidden) -> 
            # (Hidden, Heads*Dim)
            o_k = ops.reshape(o_layer.kernel, (-1, backbone.hidden_dim))
            weights_map[f"model.layers.{i}.self_attn.o_proj.weight"] = ops.transpose(o_k)

        # --- MLP (SwiGLU) ---
        # Gate (With activation)
        gate_w = decoder_layer._feedforward_gate_dense.kernel
        weights_map[f"model.layers.{i}.mlp.gate_proj.weight"] = ops.transpose(gate_w)
        
        # Up (Intermediate)
        up_w = decoder_layer._feedforward_intermediate_dense.kernel
        weights_map[f"model.layers.{i}.mlp.up_proj.weight"] = ops.transpose(up_w)
        
        # Down (Output)
        down_w = decoder_layer._feedforward_output_dense.kernel
        weights_map[f"model.layers.{i}.mlp.down_proj.weight"] = ops.transpose(down_w)

    # Final Norm
    weights_map["model.norm.weight"] = backbone.get_layer(
        "sequence_output_layernorm"
    ).scale

    # LM Head
    if include_lm_head:
        if backbone.tie_word_embeddings:
            # If tied, point to input embeddings (Exporter handles cloning)
            weights_map["lm_head.weight"] = weights_map["model.embed_tokens.weight"]
        else:
            lm_head_w = backbone.get_layer("token_embedding").reverse_embeddings
            # HF expects (Vocab, Hidden). Keras ReversibleEmbedding s
            # tores (Vocab, Hidden).
            # No transpose needed usually, but check if your version differs.
            weights_map["lm_head.weight"] = lm_head_w

    return weights_map


def get_qwen3_tokenizer_config(tokenizer):
    """Convert Keras Qwen3 tokenizer config to Hugging Face."""
    return {
        "tokenizer_class": "Qwen2Tokenizer",
        "bos_token": None,
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "unk_token": None,
        "model_max_length": 32768,
    }
    