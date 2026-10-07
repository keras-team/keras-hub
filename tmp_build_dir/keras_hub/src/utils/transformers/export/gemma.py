import keras.ops as ops


def get_gemma_config(backbone):
    token_embedding_layer = backbone.get_layer("token_embedding")
    hf_config = {
        "vocab_size": backbone.vocabulary_size,
        "num_hidden_layers": backbone.num_layers,
        "num_attention_heads": backbone.num_query_heads,
        "num_key_value_heads": backbone.num_key_value_heads,
        "hidden_size": backbone.hidden_dim,
        "intermediate_size": backbone.intermediate_dim // 2,
        "head_dim": backbone.head_dim,
        "max_position_embeddings": 8192,
        "tie_word_embeddings": token_embedding_layer.tie_weights,
        "pad_token_id": 0,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "model_type": "gemma",
    }
    return hf_config


def get_gemma_weights_map(backbone, include_lm_head=False):
    weights_dict = {}

    # Map token embedding
    token_embedding_layer = backbone.get_layer("token_embedding")
    weights_dict["model.embed_tokens.weight"] = token_embedding_layer.weights[0]

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"decoder_block_{i}")

        # Pre-attention normalization
        weights_dict[f"model.layers.{i}.input_layernorm.weight"] = (
            decoder_layer.pre_attention_norm.weights[0]
        )

        # Attention query projection
        query_kernel = decoder_layer.attention.query_dense.weights[0]
        query_kernel = ops.transpose(query_kernel, axes=(1, 0, 2))
        query_kernel = ops.reshape(query_kernel, (-1, backbone.hidden_dim))
        query_kernel = ops.transpose(query_kernel)
        weights_dict[f"model.layers.{i}.self_attn.q_proj.weight"] = query_kernel

        # Attention key projection
        key_kernel = decoder_layer.attention.key_dense.weights[0][0]
        weights_dict[f"model.layers.{i}.self_attn.k_proj.weight"] = (
            ops.transpose(key_kernel)
        )

        # Attention value projection
        value_kernel = decoder_layer.attention.value_dense.weights[0][0]
        weights_dict[f"model.layers.{i}.self_attn.v_proj.weight"] = (
            ops.transpose(value_kernel)
        )

        # Attention output projection
        out_kernel = decoder_layer.attention.output_dense.weights[0]
        out_kernel = ops.transpose(out_kernel, axes=(2, 0, 1))
        out_kernel = ops.reshape(out_kernel, (backbone.hidden_dim, -1))
        weights_dict[f"model.layers.{i}.self_attn.o_proj.weight"] = out_kernel

        # Post-attention normalization
        weights_dict[f"model.layers.{i}.post_attention_layernorm.weight"] = (
            decoder_layer.pre_ffw_norm.weights[0]
        )

        # MLP gate projection
        gate_kernel = decoder_layer.gating_ffw.weights[0]
        weights_dict[f"model.layers.{i}.mlp.gate_proj.weight"] = ops.transpose(
            gate_kernel
        )

        # MLP up projection
        up_kernel = decoder_layer.gating_ffw_2.weights[0]
        weights_dict[f"model.layers.{i}.mlp.up_proj.weight"] = ops.transpose(
            up_kernel
        )

        # MLP down projection
        down_kernel = decoder_layer.ffw_linear.weights[0]
        weights_dict[f"model.layers.{i}.mlp.down_proj.weight"] = ops.transpose(
            down_kernel
        )

    # Map final normalization
    weights_dict["model.norm.weight"] = backbone.get_layer(
        "final_normalization"
    ).weights[0]

    # Map lm_head if embeddings are not tied
    if include_lm_head and not token_embedding_layer.tie_weights:
        weights_dict["lm_head.weight"] = ops.transpose(
            token_embedding_layer.reverse_embeddings
        )
    return weights_dict


def get_gemma_tokenizer_config(tokenizer):
    tokenizer_config = {
        "tokenizer_class": "GemmaTokenizer",
        "clean_up_tokenization_spaces": False,
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "add_bos_token": True,
        "add_eos_token": False,
        "model_max_length": 8192,
    }
    # Add added_tokens_decoder
    added_tokens_decoder = {}
    special_tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "<unk>",
        "<start_of_turn>",
        "<end_of_turn>",
    ]
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
