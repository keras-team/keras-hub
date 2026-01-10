import keras.ops as ops


def get_gpt2_config(backbone):
    """Convert Keras GPT-2 config to Hugging Face GPT2Config."""
    return {
        # Core architectural dimensions
        "vocab_size": backbone.vocabulary_size,
        "n_positions": backbone.max_sequence_length,
        "n_embd": backbone.hidden_dim,
        "n_layer": backbone.num_layers,
        "n_head": backbone.num_heads,
        "n_inner": backbone.intermediate_dim,
        # Activation and regularization
        "activation_function": "gelu_new",
        "resid_pdrop": backbone.dropout,
        "embd_pdrop": backbone.dropout,
        "attn_pdrop": backbone.dropout,
        # Numerical stability and initialization
        "layer_norm_epsilon": 1e-05,
        "initializer_range": 0.02,
        # Sequence summary settings
        "summary_type": "cls_index",
        "summary_use_proj": True,
        "summary_activation": None,
        "summary_proj_to_labels": True,
        "summary_first_dropout": backbone.dropout,
        # Model behavior and special tokens
        "scale_attn_weights": True,
        "use_cache": True,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "model_type": "gpt2",
    }


def get_gpt2_weights_map(keras_model, include_lm_head=False):
    """Create a weights map for a given GPT-2 model."""
    weights_map = {}

    # Token and position embeddings
    weights_map["transformer.wte.weight"] = keras_model.get_layer(
        "token_embedding"
    ).embeddings
    weights_map["transformer.wpe.weight"] = keras_model.get_layer(
        "position_embedding"
    ).position_embeddings

    for i in range(keras_model.num_layers):
        # Attention weights
        q_w = keras_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._query_dense.kernel
        k_w = keras_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._key_dense.kernel
        v_w = keras_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._value_dense.kernel
        q_b = keras_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._query_dense.bias
        k_b = keras_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._key_dense.bias
        v_b = keras_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._value_dense.bias

        q_w = ops.reshape(q_w, (keras_model.hidden_dim, keras_model.hidden_dim))
        k_w = ops.reshape(k_w, (keras_model.hidden_dim, keras_model.hidden_dim))
        v_w = ops.reshape(v_w, (keras_model.hidden_dim, keras_model.hidden_dim))

        c_attn_w = ops.concatenate([q_w, k_w, v_w], axis=-1)
        weights_map[f"transformer.h.{i}.attn.c_attn.weight"] = c_attn_w

        q_b = ops.reshape(q_b, [-1])
        k_b = ops.reshape(k_b, [-1])
        v_b = ops.reshape(v_b, [-1])

        c_attn_b = ops.concatenate([q_b, k_b, v_b], axis=-1)
        weights_map[f"transformer.h.{i}.attn.c_attn.bias"] = c_attn_b

        # Attention projection
        c_proj_w = keras_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._output_dense.kernel
        c_proj_w = ops.reshape(
            c_proj_w, (keras_model.hidden_dim, keras_model.hidden_dim)
        )
        weights_map[f"transformer.h.{i}.attn.c_proj.weight"] = c_proj_w
        weights_map[f"transformer.h.{i}.attn.c_proj.bias"] = (
            keras_model.get_layer(
                f"transformer_layer_{i}"
            )._self_attention_layer._output_dense.bias
        )

        # Layer norms
        weights_map[f"transformer.h.{i}.ln_1.weight"] = keras_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer_norm.gamma
        weights_map[f"transformer.h.{i}.ln_1.bias"] = keras_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer_norm.beta
        weights_map[f"transformer.h.{i}.ln_2.weight"] = keras_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_layer_norm.gamma
        weights_map[f"transformer.h.{i}.ln_2.bias"] = keras_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_layer_norm.beta

        # MLP
        c_fc_w = keras_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_intermediate_dense.kernel
        weights_map[f"transformer.h.{i}.mlp.c_fc.weight"] = c_fc_w
        weights_map[f"transformer.h.{i}.mlp.c_fc.bias"] = keras_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_intermediate_dense.bias
        c_proj_w_mlp = keras_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_output_dense.kernel
        weights_map[f"transformer.h.{i}.mlp.c_proj.weight"] = c_proj_w_mlp
        weights_map[f"transformer.h.{i}.mlp.c_proj.bias"] = (
            keras_model.get_layer(
                f"transformer_layer_{i}"
            )._feedforward_output_dense.bias
        )

    # Final layer norm
    weights_map["transformer.ln_f.weight"] = keras_model.get_layer(
        "layer_norm"
    ).gamma
    weights_map["transformer.ln_f.bias"] = keras_model.get_layer(
        "layer_norm"
    ).beta

    if include_lm_head:
        # lm_head is tied to token embeddings
        weights_map["lm_head.weight"] = weights_map["transformer.wte.weight"]

    return weights_map


def get_gpt2_tokenizer_config(tokenizer):
    return {
        "model_type": "gpt2",
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
    }
