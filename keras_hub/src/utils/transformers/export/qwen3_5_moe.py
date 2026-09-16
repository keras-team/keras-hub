from keras import ops


def get_qwen3_5_moe_config(backbone):
    """Convert KerasHub Qwen3_5Moe backbone config to HF config dictionary."""

    hf_config = {
        "architectures": ["Qwen2MoeForCausalLM"],
        "model_type": "qwen2_moe",
        "vocab_size": backbone.vocabulary_size,
        "hidden_size": backbone.hidden_dim,
        "num_hidden_layers": backbone.num_layers,
        "num_attention_heads": backbone.num_query_heads,
        "num_key_value_heads": backbone.num_key_value_heads,
        "hidden_act": "silu",
        "max_position_embeddings": getattr(
            backbone, "max_sequence_length", 32768
        ),
        "rms_norm_eps": backbone.layer_norm_epsilon,
        "use_sliding_window": False,
        "tie_word_embeddings": backbone.tie_word_embeddings,
        "rope_theta": backbone.rope_max_wavelength,
        # MoE specific config
        "decoder_sparse_step": backbone.decoder_sparse_step,
        "moe_intermediate_size": backbone.moe_intermediate_dim,
        "shared_expert_intermediate_size": backbone.intermediate_dim,
        "num_experts_per_tok": backbone.top_k,
        "num_experts": backbone.num_experts,
        "norm_top_k_prob": backbone.norm_top_k_prob,
        "router_aux_loss_coef": backbone.router_aux_loss_coefficient,
    }

    return hf_config


def get_qwen3_5_moe_weights_map(backbone, include_lm_head=False):
    """Create a weights map from KerasHub Qwen3_5Moe backbone to HF format."""
    weights_map = {}

    # --- Embeddings ---
    weights_map["model.embed_tokens.weight"] = backbone.get_layer(
        "token_embedding"
    ).embeddings

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"transformer_layer_{i}")
        prefix = f"model.layers.{i}"

        # Input layernorm.
        weights_map[f"{prefix}.input_layernorm.weight"] = (
            decoder_layer._input_layernorm.scale
        )

        # Attention
        attn = decoder_layer._self_attention_layer

        # Q projection
        q_kernel = attn._query_dense.kernel
        q_kernel = ops.reshape(q_kernel, (backbone.hidden_dim, -1))
        weights_map[f"{prefix}.self_attn.q_proj.weight"] = ops.transpose(
            q_kernel
        )

        # K projection
        k_kernel = attn._key_dense.kernel
        k_kernel = ops.reshape(k_kernel, (backbone.hidden_dim, -1))
        weights_map[f"{prefix}.self_attn.k_proj.weight"] = ops.transpose(
            k_kernel
        )

        # V projection
        v_kernel = attn._value_dense.kernel
        v_kernel = ops.reshape(v_kernel, (backbone.hidden_dim, -1))
        weights_map[f"{prefix}.self_attn.v_proj.weight"] = ops.transpose(
            v_kernel
        )

        # Output projection
        o_kernel = attn._output_dense.kernel
        o_kernel = ops.reshape(o_kernel, (-1, backbone.hidden_dim))
        weights_map[f"{prefix}.self_attn.o_proj.weight"] = ops.transpose(
            o_kernel
        )

        # Post-attention layernorm.
        weights_map[f"{prefix}.post_attention_layernorm.weight"] = (
            decoder_layer._post_attention_layernorm.scale
        )

        # MLP / MoE
        if decoder_layer.is_sparse_mlp:
            moe_layer = decoder_layer._moe_layer

            # Router
            weights_map[f"{prefix}.mlp.gate.weight"] = ops.transpose(
                moe_layer._sparse_feedforward_gate_dense.kernel
            )

            # Shared expert (if exists in Qwen2Moe HF model)
            # In Keras, we have a shared MLP `_shared_expert_feedforward...`
            # Wait, Qwen3_5MoeDecoder might have a shared MLP.
            if hasattr(decoder_layer, "_shared_mlp_intermediate_dense"):
                weights_map[f"{prefix}.mlp.shared_expert.up_proj.weight"] = (
                    ops.transpose(
                        decoder_layer._shared_mlp_intermediate_dense.kernel
                    )
                )
                weights_map[f"{prefix}.mlp.shared_expert.gate_proj.weight"] = (
                    ops.transpose(decoder_layer._shared_mlp_gate_dense.kernel)
                )
                weights_map[f"{prefix}.mlp.shared_expert.down_proj.weight"] = (
                    ops.transpose(decoder_layer._shared_mlp_output_dense.kernel)
                )

            # Shared expert gate
            if hasattr(decoder_layer, "_shared_expert_gate"):
                weights_map[f"{prefix}.mlp.shared_expert_gate.weight"] = (
                    ops.transpose(decoder_layer._shared_expert_gate.kernel)
                )

            # Experts
            # Keras shape: _expert_feedforward_gate_dense is
            # (num_experts, hidden_dim, 2*intermediate_dim)
            # which we split into gate and up.
            gate_up = moe_layer.expert_bank._expert_feedforward_gate_dense
            gate, up = ops.split(gate_up, 2, axis=-1)

            down = moe_layer.expert_bank._expert_feedforward_output_dense

            for e in range(backbone.num_experts):
                weights_map[f"{prefix}.mlp.experts.{e}.gate_proj.weight"] = (
                    ops.transpose(gate[e])
                )
                weights_map[f"{prefix}.mlp.experts.{e}.up_proj.weight"] = (
                    ops.transpose(up[e])
                )
                weights_map[f"{prefix}.mlp.experts.{e}.down_proj.weight"] = (
                    ops.transpose(down[e])
                )

        else:
            # Dense MLP
            weights_map[f"{prefix}.mlp.up_proj.weight"] = ops.transpose(
                decoder_layer._feedforward_intermediate_dense.kernel
            )
            weights_map[f"{prefix}.mlp.down_proj.weight"] = ops.transpose(
                decoder_layer._feedforward_output_dense.kernel
            )
            weights_map[f"{prefix}.mlp.gate_proj.weight"] = ops.transpose(
                decoder_layer._feedforward_gate_dense.kernel
            )

    # Final normalization layer.
    weights_map["model.norm.weight"] = backbone.get_layer(
        "sequence_output_layernorm"
    ).scale

    # LM Head.
    if include_lm_head:
        token_embedding_layer = backbone.get_layer("token_embedding")
        if backbone.tie_word_embeddings:
            weights_map["lm_head.weight"] = weights_map[
                "model.embed_tokens.weight"
            ]
        else:
            weights_map["lm_head.weight"] = ops.transpose(
                token_embedding_layer.reverse_embeddings
            )

    return weights_map


def get_qwen3_5_moe_tokenizer_config(tokenizer):
    return {
        "tokenizer_class": "Qwen2Tokenizer",
        "bos_token": None,
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "unk_token": None,
        "model_max_length": 32768,
    }
