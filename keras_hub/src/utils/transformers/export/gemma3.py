import keras.ops as ops


def get_gemma3_config(backbone):
    """Convert Keras Gemma3 config to Hugging Face config dictionary."""

    layer_types = []
    for i in range(backbone.num_layers):
        if backbone.use_sliding_window_attention and (i % 6 < 5):
            layer_types.append("sliding_attention")
        else:
            layer_types.append("full_attention")

    hf_config = {
        "architectures": ["Gemma3ForCausalLM"],
        "model_type": "gemma3_text",
        "vocab_size": backbone.vocabulary_size,
        "num_hidden_layers": backbone.num_layers,
        "num_attention_heads": backbone.num_query_heads,
        "num_key_value_heads": backbone.num_key_value_heads,
        "hidden_size": backbone.hidden_dim,
        "intermediate_size": backbone.intermediate_dim,
        "head_dim": backbone.head_dim,
        "rms_norm_eps": backbone.layer_norm_epsilon,
        "rope_theta": 1000000.0,
        "attention_bias": False,
        "attention_dropout": backbone.dropout,
        "hidden_activation": "gelu_pytorch_tanh",
        # Added missing keys to match official config
        "sliding_window": backbone.sliding_window_size,
        "_sliding_window_pattern": 6,
        "use_cache": True,
        "torch_dtype": backbone.dtype_policy.name,
        "layer_types": layer_types,
        "query_pre_attn_scalar": backbone.head_dim
        if backbone.query_head_dim_normalize
        else backbone.hidden_dim // backbone.num_query_heads,
    }

    return hf_config


def get_gemma3_weights_map(backbone, include_lm_head=False):
    """Convert a Keras Gemma3 model to Hugging Face format.

    include_lm_head: If True, exports for CausalLM (with "model." prefix).
                    If False, exports for backbone only (without prefix).
    """

    def _convert_qkv_kernel(kernel, hidden_dim):
        """Helper to convert Q/K/V projection kernels to HF format.

        Args:
            kernel: The kernel weight tensor to convert.
            hidden_dim: The hidden dimension size for reshaping.

        Returns:
            Converted kernel in HF format.
        """
        kernel = ops.transpose(kernel, axes=(1, 0, 2))  # permute(1, 0, 2)
        kernel = ops.reshape(kernel, (hidden_dim, -1))
        kernel = ops.transpose(kernel)  # .T
        return kernel

    weights_dict = {}

    # For CausalLM export, use "model." prefix
    # For backbone export, use no prefix
    prefix = "model." if include_lm_head else ""

    # Token embeddings - use .weights[0] to get backend tensor
    token_embedding_layer = backbone.get_layer("token_embedding")
    token_embedding = token_embedding_layer.weights[0]
    weights_dict[f"{prefix}embed_tokens.weight"] = token_embedding

    for i in range(backbone.num_layers):
        block = backbone.get_layer(f"decoder_block_{i}")

        # Attention query projection
        q_kernel = _convert_qkv_kernel(
            block.attention.query_dense.weights[0], backbone.hidden_dim
        )
        weights_dict[f"{prefix}layers.{i}.self_attn.q_proj.weight"] = q_kernel

        # Attention key projection
        k_kernel = _convert_qkv_kernel(
            block.attention.key_dense.weights[0], backbone.hidden_dim
        )
        weights_dict[f"{prefix}layers.{i}.self_attn.k_proj.weight"] = k_kernel

        # Attention value projection
        v_kernel = _convert_qkv_kernel(
            block.attention.value_dense.weights[0], backbone.hidden_dim
        )
        weights_dict[f"{prefix}layers.{i}.self_attn.v_proj.weight"] = v_kernel

        # Attention output projection
        o_kernel = block.attention.output_dense.weights[0]
        o_kernel = ops.transpose(o_kernel, axes=(2, 0, 1))  # permute(2, 0, 1)
        o_kernel = ops.reshape(o_kernel, (backbone.hidden_dim, -1))
        weights_dict[f"{prefix}layers.{i}.self_attn.o_proj.weight"] = o_kernel

        # Query and key normalization
        q_norm = block.attention.query_norm.weights[0]
        weights_dict[f"{prefix}layers.{i}.self_attn.q_norm.weight"] = q_norm

        k_norm = block.attention.key_norm.weights[0]
        weights_dict[f"{prefix}layers.{i}.self_attn.k_norm.weight"] = k_norm

        # MLP gate projection
        gate_kernel = block.gating_ffw.weights[0]
        gate_kernel = ops.transpose(gate_kernel)  # .T
        weights_dict[f"{prefix}layers.{i}.mlp.gate_proj.weight"] = gate_kernel

        # MLP up projection
        up_kernel = block.gating_ffw_2.weights[0]
        up_kernel = ops.transpose(up_kernel)  # .T
        weights_dict[f"{prefix}layers.{i}.mlp.up_proj.weight"] = up_kernel

        # MLP down projection
        down_kernel = block.ffw_linear.weights[0]
        down_kernel = ops.transpose(down_kernel)  # .T
        weights_dict[f"{prefix}layers.{i}.mlp.down_proj.weight"] = down_kernel

        # Pre-attention normalization
        input_layer_norm = block.pre_attention_norm.weights[0]
        weights_dict[f"{prefix}layers.{i}.input_layernorm.weight"] = (
            input_layer_norm
        )

        # Post-attention normalization
        if hasattr(block, "post_attention_norm"):
            post_attn_norm = block.post_attention_norm.weights[0]
            weights_dict[
                f"{prefix}layers.{i}.post_attention_layernorm.weight"
            ] = post_attn_norm
        # Pre-feedforward normalization
        pre_feedforward_layernorm = block.pre_ffw_norm.weights[0]
        weights_dict[f"{prefix}layers.{i}.pre_feedforward_layernorm.weight"] = (
            pre_feedforward_layernorm
        )
        # Post-feedforward normalization
        if hasattr(block, "post_ffw_norm"):
            post_feedforward_layernorm = block.post_ffw_norm.weights[0]
            weights_dict[
                f"{prefix}layers.{i}.post_feedforward_layernorm.weight"
            ] = post_feedforward_layernorm

    # Final normalization
    final_norm = backbone.get_layer("final_normalization").weights[0]
    weights_dict[f"{prefix}norm.weight"] = final_norm

    if include_lm_head and not token_embedding_layer.tie_weights:
        weights_dict["lm_head.weight"] = ops.transpose(
            token_embedding_layer.reverse_embeddings
        )

    return weights_dict


def get_gemma3_tokenizer_config(tokenizer):
    tokenizer_config = {
        "tokenizer_class": "GemmaTokenizer",
        "clean_up_tokenization_spaces": False,
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "add_bos_token": True,
        "add_eos_token": False,
        "model_max_length": 1000000000000000019884624838656,
    }
    # Add added_tokens_decoder
    added_tokens_decoder = {}
    special_tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "<unk>",
        "<mask>",
        "[multimodal]",
        "<img>",
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
