import keras.ops as ops


def get_llama_config(backbone):
    token_embedding_layer = backbone.get_layer("token_embedding")
    hf_config = {
        "vocab_size": backbone.vocabulary_size,
        "num_hidden_layers": backbone.num_layers,
        "num_attention_heads": backbone.num_query_heads,
        "num_key_value_heads": backbone.num_key_value_heads,
        "hidden_size": backbone.hidden_dim,
        "intermediate_size": backbone.intermediate_dim,
        "max_position_embeddings": 4096,
        "tie_word_embeddings": token_embedding_layer.tie_weights,
        "rms_norm_eps": backbone.layer_norm_epsilon,
        "rope_theta": backbone.rope_max_wavelength,
        "model_type": "llama",
    }
    return hf_config


def get_llama_weights_map(backbone, include_lm_head=False):
    weights_dict = {}
    # Map token embedding
    token_embedding_layer = backbone.token_embedding
    weights_dict["model.embed_tokens.weight"] = token_embedding_layer.embeddings
    for i in range(backbone.num_layers):
        transformer_layer = backbone.transformer_layers[i]
        # Pre-attention normalization
        weights_dict[f"model.layers.{i}.input_layernorm.weight"] = (
            transformer_layer._self_attention_layernorm.weights[0]
        )
        # Attention query projection
        query_kernel = (
            transformer_layer._self_attention_layer._query_dense.weights[0]
        )
        query_kernel = ops.reshape(query_kernel, (backbone.hidden_dim, -1))
        query_kernel = ops.transpose(query_kernel)
        weights_dict[f"model.layers.{i}.self_attn.q_proj.weight"] = query_kernel
        # Attention key projection
        key_kernel = transformer_layer._self_attention_layer._key_dense.weights[
            0
        ]
        key_kernel = ops.reshape(key_kernel, (backbone.hidden_dim, -1))
        key_kernel = ops.transpose(key_kernel)
        weights_dict[f"model.layers.{i}.self_attn.k_proj.weight"] = key_kernel
        # Attention value projection
        value_kernel = (
            transformer_layer._self_attention_layer._value_dense.weights[0]
        )
        value_kernel = ops.reshape(value_kernel, (backbone.hidden_dim, -1))
        value_kernel = ops.transpose(value_kernel)
        weights_dict[f"model.layers.{i}.self_attn.v_proj.weight"] = value_kernel
        # Attention output projection
        out_kernel = (
            transformer_layer._self_attention_layer._output_dense.weights[0]
        )
        out_kernel = ops.reshape(out_kernel, (-1, backbone.hidden_dim))
        out_kernel = ops.transpose(out_kernel)
        weights_dict[f"model.layers.{i}.self_attn.o_proj.weight"] = out_kernel
        # Post-attention normalization
        weights_dict[f"model.layers.{i}.post_attention_layernorm.weight"] = (
            transformer_layer._feedforward_layernorm.weights[0]
        )
        # MLP gate projection
        gate_kernel = transformer_layer._feedforward_gate_dense.weights[0]
        weights_dict[f"model.layers.{i}.mlp.gate_proj.weight"] = ops.transpose(
            gate_kernel
        )
        # MLP up projection
        up_kernel = transformer_layer._feedforward_intermediate_dense.weights[0]
        weights_dict[f"model.layers.{i}.mlp.up_proj.weight"] = ops.transpose(
            up_kernel
        )
        # MLP down projection
        down_kernel = transformer_layer._feedforward_output_dense.weights[0]
        weights_dict[f"model.layers.{i}.mlp.down_proj.weight"] = ops.transpose(
            down_kernel
        )
    # Map final normalization
    weights_dict["model.norm.weight"] = backbone.layer_norm.weights[0]
    # Map lm head
    if include_lm_head and not token_embedding_layer.tie_weights:
        weights_dict["lm_head.weight"] = ops.transpose(
            token_embedding_layer.reverse_embeddings
        )
    return weights_dict


def get_llama_tokenizer_config(tokenizer):
    tokenizer_config = {
        "tokenizer_class": "LlamaTokenizer",
        "clean_up_tokenization_spaces": False,
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "add_bos_token": True,
        "add_eos_token": False,
        "model_max_length": 4096,
    }
    # Add added_tokens_decoder
    added_tokens_decoder = {}
    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>", "<s>", "</s>"]
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
