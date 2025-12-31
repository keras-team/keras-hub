import keras.ops as ops
import transformers

from keras_hub.src.models.qwen_moe.qwen_moe_decoder import QwenSparseMoeBlock


def get_qwen_moe_config(backbone):
    """Convert Keras Qwen MoE config to Hugging Face Qwen2MoeConfig."""
    return transformers.Qwen2MoeConfig(
        vocab_size=backbone.vocabulary_size,
        hidden_size=backbone.hidden_dim,
        num_hidden_layers=backbone.num_layers,
        num_attention_heads=backbone.num_query_heads,
        num_key_value_heads=backbone.num_key_value_heads,
        # MoE Specifics
        num_experts=backbone.num_experts,
        num_experts_per_tok=backbone.top_k,
        moe_intermediate_size=backbone.moe_intermediate_dim,
        shared_expert_intermediate_size=backbone.shared_expert_intermediate_dim,
        # Standard Qwen settings
        hidden_act="silu",
        rms_norm_eps=backbone.layer_norm_epsilon,
        rope_theta=backbone.rope_max_wavelength,
        tie_word_embeddings=backbone.tie_word_embeddings,
        initializer_range=0.02,
        use_cache=True,
    )


def get_qwen_moe_weights_map(backbone, include_lm_head=False):
    """Create a weights map for a given Qwen MoE model."""
    weights_map = {}

    weights_map["model.embed_tokens.weight"] = backbone.get_layer(
        "token_embedding"
    ).embeddings

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"transformer_layer_{i}")

        weights_map[f"model.layers.{i}.input_layernorm.weight"] = (
            decoder_layer._self_attention_layernorm.scale
        )
        weights_map[f"model.layers.{i}.post_attention_layernorm.weight"] = (
            decoder_layer._feedforward_layernorm.scale
        )

        # --- Attention ---
        attn_layer = decoder_layer._self_attention_layer

        # Keras Dense: (Hidden, Heads, Dim) -> HF Linear (Out, In)
        def map_attn_linear(keras_layer, hf_name, bias=True):
            # Kernel: (Hidden, Heads, Dim) -> (Hidden, Heads*Dim) ->
            # (Heads*Dim, Hidden)
            k = ops.reshape(keras_layer.kernel, (backbone.hidden_dim, -1))
            weights_map[f"model.layers.{i}.self_attn.{hf_name}.weight"] = (
                ops.transpose(k)
            )

            if bias:
                # Bias: (Heads, Dim) -> (Heads*Dim)
                b = ops.reshape(keras_layer.bias, (-1,))
                weights_map[f"model.layers.{i}.self_attn.{hf_name}.bias"] = b

        map_attn_linear(attn_layer.query_dense, "q_proj", bias=True)
        map_attn_linear(attn_layer.key_dense, "k_proj", bias=True)
        map_attn_linear(attn_layer.value_dense, "v_proj", bias=True)

        # Kernel: (Heads, Dim, Hidden) -> (Heads*Dim, Hidden) ->
        # (Hidden, Heads*Dim)
        o_k = ops.reshape(
            attn_layer._output_dense.kernel, (-1, backbone.hidden_dim)
        )
        weights_map[f"model.layers.{i}.self_attn.o_proj.weight"] = (
            ops.transpose(o_k)
        )

        # --- MLP / MOE BLOCK ---
        mlp_layer = decoder_layer.mlp

        if isinstance(mlp_layer, QwenSparseMoeBlock):
            # === MOE LAYER ===

            # 1. Router (Gate)
            router_w = mlp_layer._sparse_feedforward_gate_dense.kernel
            weights_map[f"model.layers.{i}.block_sparse_moe.gate.weight"] = (
                ops.transpose(router_w)
            )

            # 2. Shared Expert (Standard SwiGLU)
            shared = mlp_layer.shared_expert_dense
            # Gate (In, Intermed) -> (Intermed, In)
            weights_map[
                f"model.layers.{i}.block_sparse_moe.shared_expert.gate_proj.weight"
            ] = ops.transpose(shared._feedforward_gate_dense.kernel)
            # Up (In, Intermed) -> (Intermed, In)
            weights_map[
                f"model.layers.{i}.block_sparse_moe.shared_expert.up_proj.weight"
            ] = ops.transpose(shared._feedforward_intermediate_dense.kernel)
            # Down (Intermed, In) -> (In, Intermed)
            weights_map[
                f"model.layers.{i}.block_sparse_moe.shared_expert.down_proj.weight"
            ] = ops.transpose(shared._feedforward_output_dense.kernel)

            # 3. Routed Experts (Fused Gate/Up)
            experts_fused = mlp_layer.expert_bank._expert_feedforward_gate_dense
            experts_down = (
                mlp_layer.expert_bank._expert_feedforward_output_dense
            )

            for e in range(backbone.num_experts):
                expert_slice = experts_fused[e, :, :]

                # Split into Gate and Up along last axis
                gate_expert, up_expert = ops.split(expert_slice, 2, axis=-1)

                # Transpose for HF Linear: (Hidden, Intermed) ->
                # (Intermed, Hidden)
                weights_map[
                    f"model.layers.{i}.block_sparse_moe.experts.{e}.gate_proj.weight"
                ] = ops.transpose(gate_expert)
                weights_map[
                    f"model.layers.{i}.block_sparse_moe.experts.{e}.up_proj.weight"
                ] = ops.transpose(up_expert)

                down_expert = experts_down[e, :, :]
                # Transpose: (Hidden, Intermed)
                weights_map[
                    f"model.layers.{i}.block_sparse_moe.experts.{e}.down_proj.weight"
                ] = ops.transpose(down_expert)

        else:
            # Standard MLP layer mapping if not sparse
            weights_map[f"model.layers.{i}.mlp.gate_proj.weight"] = (
                ops.transpose(mlp_layer._feedforward_gate_dense.kernel)
            )
            weights_map[f"model.layers.{i}.mlp.up_proj.weight"] = ops.transpose(
                mlp_layer._feedforward_intermediate_dense.kernel
            )
            weights_map[f"model.layers.{i}.mlp.down_proj.weight"] = (
                ops.transpose(mlp_layer._feedforward_output_dense.kernel)
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
            lm_head = backbone.get_layer("token_embedding").reverse_embeddings
            weights_map["lm_head.weight"] = ops.transpose(lm_head)

    return weights_map
