"""Export utilities for Qwen3.5 models to HuggingFace Transformers format.

Provides three functions following the KerasHub export pattern:
  - get_qwen3_5_config: Backbone config → HF config dict
  - get_qwen3_5_weights_map: Backbone weights → HF weight name mapping
  - get_qwen3_5_tokenizer_config: Tokenizer → HF tokenizer config dict
"""

import keras.ops as ops


def get_qwen3_5_config(backbone):
    """Convert KerasHub Qwen3.5 backbone config to HF config dictionary."""

    # Build rope_parameters to match HF's nested structure.
    rope_parameters = {
        "rope_type": "default",
        "rope_theta": backbone.rope_max_wavelength,
        "partial_rotary_factor": backbone.partial_rotary_factor,
    }
    if backbone.mrope_section is not None:
        rope_parameters["mrope_section"] = backbone.mrope_section
        rope_parameters["mrope_interleaved"] = True

    # Build text_config (nested under the top-level config in HF).
    text_config = {
        "model_type": "qwen3_5_text",
        "vocab_size": backbone.vocabulary_size,
        "hidden_size": backbone.hidden_dim,
        "num_hidden_layers": backbone.num_layers,
        "num_attention_heads": backbone.num_query_heads,
        "num_key_value_heads": backbone.num_key_value_heads,
        "intermediate_size": backbone.intermediate_dim,
        "head_dim": backbone.head_dim,
        "rms_norm_eps": backbone.layer_norm_epsilon,
        "hidden_act": "silu",
        "use_cache": True,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "tie_word_embeddings": backbone.tie_word_embeddings,
        "rope_parameters": rope_parameters,
        "layer_types": backbone.layer_types,
        "linear_num_key_heads": backbone.linear_num_key_heads,
        "linear_num_value_heads": backbone.linear_num_value_heads,
        "linear_key_head_dim": backbone.linear_key_head_dim,
        "linear_value_head_dim": backbone.linear_value_head_dim,
        "linear_conv_kernel_dim": backbone.linear_conv_kernel_dim,
        "mamba_ssm_dtype": "float32",
    }

    # Add max_position_embeddings if available.
    if hasattr(backbone, "max_sequence_length"):
        text_config["max_position_embeddings"] = backbone.max_sequence_length

    hf_config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "tie_word_embeddings": backbone.tie_word_embeddings,
        "text_config": text_config,
    }

    # Add vision_config if the backbone has a vision encoder.
    if (
        hasattr(backbone, "vision_encoder")
        and backbone.vision_encoder is not None
    ):
        vis = backbone.vision_encoder
        hf_config["vision_config"] = {
            "model_type": "qwen3_5",
            "depth": vis.depth,
            "hidden_size": vis.hidden_size,
            "num_heads": vis.num_heads,
            "intermediate_size": vis.intermediate_size,
            "in_channels": vis.in_channels,
            "patch_size": vis.patch_size,
            "temporal_patch_size": vis.temporal_patch_size,
            "spatial_merge_size": vis.spatial_merge_size,
            "out_hidden_size": vis.out_hidden_size,
            "num_position_embeddings": vis.num_position_embeddings,
            "hidden_act": "gelu_pytorch_tanh",
        }

        # Add special token IDs if the backbone stores them.
        if hasattr(backbone, "image_token_id"):
            hf_config["image_token_id"] = backbone.image_token_id
        if hasattr(backbone, "video_token_id"):
            hf_config["video_token_id"] = backbone.video_token_id

    return hf_config


def get_qwen3_5_weights_map(backbone, include_lm_head=False):
    """Create a weights map from KerasHub Qwen3.5 backbone to HF format.

    This is the inverse of the import converter in convert_qwen3_5.py.
    Each HF weight key maps to the corresponding Keras tensor, with
    appropriate transposes to reverse the import hook_fns.
    """
    weights_map = {}

    # --- Embeddings ---
    weights_map["model.language_model.embed_tokens.weight"] = (
        backbone.get_layer("token_embedding").embeddings
    )

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"transformer_layer_{i}")
        layer_type = decoder_layer.layer_type
        prefix = f"model.language_model.layers.{i}"

        # Input layernorm.
        weights_map[f"{prefix}.input_layernorm.weight"] = (
            decoder_layer._input_layernorm.scale
        )

        if layer_type == "full_attention":
            attn = decoder_layer._self_attention_layer

            # Q projection: Keras shape (hidden_dim, num_heads, head_dim*2)
            # → HF shape (num_heads * head_dim * 2, hidden_dim), i.e. transpose
            q_kernel = attn._query_dense.kernel
            q_kernel = ops.reshape(q_kernel, (backbone.hidden_dim, -1))
            weights_map[f"{prefix}.self_attn.q_proj.weight"] = ops.transpose(
                q_kernel
            )

            # Q norm.
            weights_map[f"{prefix}.self_attn.q_norm.weight"] = (
                attn._query_norm.scale
            )

            # K projection: Keras shape (hidden_dim, num_kv_heads, head_dim)
            # → HF shape (num_kv_heads * head_dim, hidden_dim)
            k_kernel = attn._key_dense.kernel
            k_kernel = ops.reshape(k_kernel, (backbone.hidden_dim, -1))
            weights_map[f"{prefix}.self_attn.k_proj.weight"] = ops.transpose(
                k_kernel
            )

            # K norm.
            weights_map[f"{prefix}.self_attn.k_norm.weight"] = (
                attn._key_norm.scale
            )

            # V projection.
            v_kernel = attn._value_dense.kernel
            v_kernel = ops.reshape(v_kernel, (backbone.hidden_dim, -1))
            weights_map[f"{prefix}.self_attn.v_proj.weight"] = ops.transpose(
                v_kernel
            )

            # Output projection: Keras shape (num_heads, head_dim, hidden_dim)
            # → HF shape (hidden_dim, num_heads * head_dim)
            o_kernel = attn._output_dense.kernel
            o_kernel = ops.reshape(o_kernel, (-1, backbone.hidden_dim))
            weights_map[f"{prefix}.self_attn.o_proj.weight"] = ops.transpose(
                o_kernel
            )

        elif layer_type == "linear_attention":
            gdn = decoder_layer._linear_attn

            # in_proj_qkv: Keras (hidden_dim, qkv_dim) → HF (qkv_dim, hidden)
            weights_map[f"{prefix}.linear_attn.in_proj_qkv.weight"] = (
                ops.transpose(gdn.in_proj_qkv.kernel)
            )

            # in_proj_z.
            weights_map[f"{prefix}.linear_attn.in_proj_z.weight"] = (
                ops.transpose(gdn.in_proj_z.kernel)
            )

            # in_proj_b (write gate).
            weights_map[f"{prefix}.linear_attn.in_proj_b.weight"] = (
                ops.transpose(gdn.in_proj_b.kernel)
            )

            # in_proj_a (decay gate).
            weights_map[f"{prefix}.linear_attn.in_proj_a.weight"] = (
                ops.transpose(gdn.in_proj_a.kernel)
            )

            # Conv1d weight: Keras (channels, kernel_size)
            # → HF (channels, 1, kernel_size)
            weights_map[f"{prefix}.linear_attn.conv1d.weight"] = (
                ops.expand_dims(gdn.conv1d_weight, axis=1)
            )

            # dt_bias.
            weights_map[f"{prefix}.linear_attn.dt_bias"] = gdn.dt_bias

            # A_log.
            weights_map[f"{prefix}.linear_attn.A_log"] = gdn.A_log

            # Output gated RMSNorm.
            weights_map[f"{prefix}.linear_attn.norm.weight"] = gdn.norm.scale

            # Output projection.
            weights_map[f"{prefix}.linear_attn.out_proj.weight"] = (
                ops.transpose(gdn.out_proj.kernel)
            )

        # MLP layers (same for both layer types).
        weights_map[f"{prefix}.mlp.up_proj.weight"] = ops.transpose(
            decoder_layer._feedforward_intermediate_dense.kernel
        )
        weights_map[f"{prefix}.mlp.down_proj.weight"] = ops.transpose(
            decoder_layer._feedforward_output_dense.kernel
        )
        weights_map[f"{prefix}.mlp.gate_proj.weight"] = ops.transpose(
            decoder_layer._feedforward_gate_dense.kernel
        )

        # Post-attention layernorm.
        weights_map[f"{prefix}.post_attention_layernorm.weight"] = (
            decoder_layer._post_attention_layernorm.scale
        )

    # Final normalization layer.
    weights_map["model.language_model.norm.weight"] = backbone.get_layer(
        "sequence_output_layernorm"
    ).scale

    # LM Head.
    if include_lm_head:
        token_embedding_layer = backbone.get_layer("token_embedding")
        if backbone.tie_word_embeddings:
            weights_map["lm_head.weight"] = weights_map[
                "model.language_model.embed_tokens.weight"
            ]
        else:
            weights_map["lm_head.weight"] = ops.transpose(
                token_embedding_layer.reverse_embeddings
            )

    # --- Vision encoder weights (optional) ---
    if (
        hasattr(backbone, "vision_encoder")
        and backbone.vision_encoder is not None
    ):
        vis = backbone.vision_encoder
        vis_prefix = "model.visual"

        # Ensure all vision sublayers are built (they use lazy build).
        if not vis.patch_embed.built:
            vis.patch_embed.build(
                (
                    None,
                    vis.temporal_patch_size,
                    vis.patch_size,
                    vis.patch_size,
                    vis.in_channels,
                )
            )
        if not vis.pos_embed.built:
            vis.pos_embed.build((None,))
        for blk in vis.blocks:
            if not blk.built:
                blk.build((None, vis.hidden_size))
        if not vis.merger.built:
            vis.merger.build((None, vis.hidden_size))

        # Patch embedding Conv3D:
        # Keras: (temporal, patch, patch, in_channels, hidden_size)
        # HF: (hidden_size, in_channels, temporal, patch, patch)
        weights_map[f"{vis_prefix}.patch_embed.proj.weight"] = ops.transpose(
            vis.patch_embed.proj.kernel, (4, 3, 0, 1, 2)
        )
        weights_map[f"{vis_prefix}.patch_embed.proj.bias"] = (
            vis.patch_embed.proj.bias
        )

        # Absolute position embedding.
        weights_map[f"{vis_prefix}.pos_embed.weight"] = vis.pos_embed.embeddings

        # ViT blocks.
        for blk_i in range(vis.depth):
            blk = vis.blocks[blk_i]
            blk_prefix = f"{vis_prefix}.blocks.{blk_i}"

            # LayerNorm 1 & 2.
            weights_map[f"{blk_prefix}.norm1.weight"] = blk.norm1.gamma
            weights_map[f"{blk_prefix}.norm1.bias"] = blk.norm1.beta
            weights_map[f"{blk_prefix}.norm2.weight"] = blk.norm2.gamma
            weights_map[f"{blk_prefix}.norm2.bias"] = blk.norm2.beta

            # Attention QKV (fused) and output projection.
            weights_map[f"{blk_prefix}.attn.qkv.weight"] = ops.transpose(
                blk.attn.qkv.kernel
            )
            weights_map[f"{blk_prefix}.attn.qkv.bias"] = blk.attn.qkv.bias
            weights_map[f"{blk_prefix}.attn.proj.weight"] = ops.transpose(
                blk.attn.proj.kernel
            )
            weights_map[f"{blk_prefix}.attn.proj.bias"] = blk.attn.proj.bias

            # MLP fc1 and fc2.
            weights_map[f"{blk_prefix}.mlp.linear_fc1.weight"] = ops.transpose(
                blk.mlp.fc1.kernel
            )
            weights_map[f"{blk_prefix}.mlp.linear_fc1.bias"] = blk.mlp.fc1.bias
            weights_map[f"{blk_prefix}.mlp.linear_fc2.weight"] = ops.transpose(
                blk.mlp.fc2.kernel
            )
            weights_map[f"{blk_prefix}.mlp.linear_fc2.bias"] = blk.mlp.fc2.bias

        # Patch merger.
        merger_prefix = f"{vis_prefix}.merger"
        weights_map[f"{merger_prefix}.norm.weight"] = vis.merger.norm.gamma
        weights_map[f"{merger_prefix}.norm.bias"] = vis.merger.norm.beta
        weights_map[f"{merger_prefix}.linear_fc1.weight"] = ops.transpose(
            vis.merger.fc1.kernel
        )
        weights_map[f"{merger_prefix}.linear_fc1.bias"] = vis.merger.fc1.bias
        weights_map[f"{merger_prefix}.linear_fc2.weight"] = ops.transpose(
            vis.merger.fc2.kernel
        )
        weights_map[f"{merger_prefix}.linear_fc2.bias"] = vis.merger.fc2.bias

    return weights_map


def get_qwen3_5_tokenizer_config(tokenizer):
    """Convert KerasHub Qwen3.5 tokenizer config to HF format."""
    return {
        "tokenizer_class": "Qwen2Tokenizer",
        "bos_token": None,
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "unk_token": None,
        "model_max_length": 131072,
    }
