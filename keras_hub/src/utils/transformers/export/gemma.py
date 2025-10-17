import numpy as np


def get_gemma_config(backbone, include_lm_head=False):
    token_embedding_layer = backbone.get_layer("token_embedding")
    if include_lm_head:
        architectures = ["GemmaForCausalLM"]  # Full model with LM head
    else:
        architectures = ["GemmaForBackbone"]  # Just backbone
    hf_config = {
        "architectures": architectures,
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
    # Map token embedding
    token_embedding_layer = backbone.get_layer("token_embedding")
    yield "model.embed_tokens.weight", token_embedding_layer.weights[0]
    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"decoder_block_{i}")
        # Pre-attention normalization
        yield (
            f"model.layers.{i}.input_layernorm.weight",
            decoder_layer.pre_attention_norm.weights[0],
        )
        # Attention query projection
        query_kernel = decoder_layer.attention.query_dense.weights[0]
        yield f"model.layers.{i}.self_attn.q_proj.weight", query_kernel
        # Attention key projection
        key_kernel = decoder_layer.attention.key_dense.weights[0]
        yield f"model.layers.{i}.self_attn.k_proj.weight", key_kernel
        # Attention value projection
        value_kernel = decoder_layer.attention.value_dense.weights[0]
        yield f"model.layers.{i}.self_attn.v_proj.weight", value_kernel
        # Attention output projection
        out_kernel = decoder_layer.attention.output_dense.weights[0]
        yield f"model.layers.{i}.self_attn.o_proj.weight", out_kernel
        # Post-attention normalization
        yield (
            f"model.layers.{i}.post_attention_layernorm.weight",
            decoder_layer.pre_ffw_norm.weights[0],
        )
        # MLP gate projection
        gate_kernel = decoder_layer.gating_ffw.weights[0]
        yield f"model.layers.{i}.mlp.gate_proj.weight", gate_kernel
        # MLP up projection
        up_kernel = decoder_layer.gating_ffw_2.weights[0]
        yield f"model.layers.{i}.mlp.up_proj.weight", up_kernel
        # MLP down projection
        down_kernel = decoder_layer.ffw_linear.weights[0]
        yield f"model.layers.{i}.mlp.down_proj.weight", down_kernel
    # Map final normalization
    yield (
        "model.norm.weight",
        backbone.get_layer("final_normalization").weights[0],
    )
    # Map lm_head if embeddings are not tied
    if include_lm_head and not token_embedding_layer.tie_weights:
        lm_head = token_embedding_layer.reverse_embeddings
        yield "lm_head.weight", lm_head


def get_gemma_transform_fn(backbone):
    """Return a transform function for Gemma weights."""

    def transform(name, np_tensor):
        if name.endswith("q_proj.weight"):
            np_tensor = np.transpose(np_tensor, axes=(1, 0, 2))
            np_tensor = np.reshape(np_tensor, (-1, backbone.hidden_dim))
            np_tensor = np.transpose(np_tensor)
        elif name.endswith("k_proj.weight") or name.endswith("v_proj.weight"):
            np_tensor = np.transpose(np_tensor, axes=(0, 2, 1))
            np_tensor = np.reshape(np_tensor, (-1, backbone.hidden_dim))
        elif name.endswith("o_proj.weight"):
            np_tensor = np.transpose(np_tensor, axes=(2, 0, 1))
            np_tensor = np.reshape(np_tensor, (backbone.hidden_dim, -1))
        elif (
            name.endswith("gate_proj.weight")
            or name.endswith("up_proj.weight")
            or name.endswith("down_proj.weight")
        ):
            np_tensor = np.transpose(np_tensor)
        elif name == "lm_head.weight":
            np_tensor = np.transpose(np_tensor)
        return np_tensor

    return transform


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
