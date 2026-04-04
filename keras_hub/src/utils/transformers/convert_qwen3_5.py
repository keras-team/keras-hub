"""HF -> KerasHub weight converter for Qwen3.5."""

import numpy as np

from keras_hub.src.models.qwen3_5.qwen3_5_backbone import Qwen3_5Backbone
from keras_hub.src.models.qwen3_5.qwen3_5_vision_encoder import (
    Qwen3_5VisionEncoder,
)
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = Qwen3_5Backbone


def convert_backbone_config(transformers_config):
    # tie_word_embeddings is at the top-level config.
    tie_word_embeddings = transformers_config["tie_word_embeddings"]

    # Save top-level fields that live outside text_config.
    top_level_vision_config = transformers_config.get("vision_config", None)
    top_level_mrope_section = transformers_config.get("mrope_section", None)
    top_level_hidden_size = transformers_config.get("hidden_size", None)

    # Qwen3.5 text config is nested under "text_config".
    if "text_config" in transformers_config:
        transformers_config = transformers_config["text_config"]

    # rope_theta and partial_rotary_factor are nested under
    # rope_parameters in the HF config.
    rope_params = transformers_config["rope_parameters"]

    # Build layer_types list.
    num_layers = transformers_config["num_hidden_layers"]
    layer_types = transformers_config.get("layer_types", None)
    if layer_types is None:
        # Default: every 4th layer is full_attention.
        layer_types = [
            ("linear_attention" if bool((i + 1) % 4) else "full_attention")
            for i in range(num_layers)
        ]

    # M-RoPE section configuration (top-level, not inside text_config).
    mrope_section = top_level_mrope_section

    # Vision encoder configuration (top-level, optional).
    vision_encoder = None
    vision_config = top_level_vision_config
    if vision_config is not None:
        # out_hidden_size should match the text backbone's hidden_size.
        text_hidden = transformers_config.get(
            "hidden_size", top_level_hidden_size
        )
        vision_encoder = Qwen3_5VisionEncoder(
            depth=vision_config.get("depth", 24),
            hidden_size=vision_config.get("hidden_size", 1280),
            num_heads=vision_config.get("num_heads", 16),
            intermediate_size=vision_config.get("intermediate_size", 5120),
            in_channels=vision_config.get("in_channels", 3),
            patch_size=vision_config.get("patch_size", 16),
            temporal_patch_size=vision_config.get("temporal_patch_size", 2),
            spatial_merge_size=vision_config.get("spatial_merge_size", 2),
            out_hidden_size=text_hidden
            or vision_config.get("out_hidden_size", 1280),
            num_position_embeddings=vision_config.get(
                "num_position_embeddings", 2304
            ),
        )

    result = {
        "vocabulary_size": transformers_config["vocab_size"],
        "head_dim": transformers_config["head_dim"],
        "hidden_dim": transformers_config["hidden_size"],
        "num_layers": num_layers,
        "num_query_heads": transformers_config["num_attention_heads"],
        "num_key_value_heads": transformers_config["num_key_value_heads"],
        "intermediate_dim": transformers_config["intermediate_size"],
        "layer_norm_epsilon": transformers_config["rms_norm_eps"],
        "rope_max_wavelength": rope_params["rope_theta"],
        "partial_rotary_factor": rope_params["partial_rotary_factor"],
        "tie_word_embeddings": tie_word_embeddings,
        "layer_types": layer_types,
        "linear_num_key_heads": transformers_config["linear_num_key_heads"],
        "linear_num_value_heads": transformers_config["linear_num_value_heads"],
        "linear_key_head_dim": transformers_config["linear_key_head_dim"],
        "linear_value_head_dim": transformers_config["linear_value_head_dim"],
        "linear_conv_kernel_dim": transformers_config["linear_conv_kernel_dim"],
    }
    if vision_encoder is not None:
        result["vision_encoder"] = vision_encoder
    if mrope_section is not None:
        result["mrope_section"] = mrope_section
    return result


def convert_weights(backbone, loader, transformers_config):
    # Embedding.
    loader.port_weight(
        keras_variable=backbone.get_layer("token_embedding").embeddings,
        hf_weight_key="model.embed_tokens.weight",
    )
    if not backbone.tie_word_embeddings:
        loader.port_weight(
            keras_variable=backbone.get_layer(
                "token_embedding"
            ).reverse_embeddings,
            hf_weight_key="lm_head.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )

    def transpose_and_reshape(x, shape):
        return np.reshape(np.transpose(x), shape)

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"transformer_layer_{i}")
        layer_type = decoder_layer.layer_type
        prefix = f"model.layers.{i}"

        # Input layernorm.
        loader.port_weight(
            keras_variable=decoder_layer._input_layernorm.scale,
            hf_weight_key=f"{prefix}.input_layernorm.weight",
        )

        if layer_type == "full_attention":
            attn = decoder_layer._self_attention_layer

            # Q projection (includes gate: head_dim * 2).
            loader.port_weight(
                keras_variable=attn._query_dense.kernel,
                hf_weight_key=f"{prefix}.self_attn.q_proj.weight",
                hook_fn=transpose_and_reshape,
            )
            # Q norm.
            loader.port_weight(
                keras_variable=attn._query_norm.scale,
                hf_weight_key=f"{prefix}.self_attn.q_norm.weight",
            )
            # K projection.
            loader.port_weight(
                keras_variable=attn._key_dense.kernel,
                hf_weight_key=f"{prefix}.self_attn.k_proj.weight",
                hook_fn=transpose_and_reshape,
            )
            # K norm.
            loader.port_weight(
                keras_variable=attn._key_norm.scale,
                hf_weight_key=f"{prefix}.self_attn.k_norm.weight",
            )
            # V projection.
            loader.port_weight(
                keras_variable=attn._value_dense.kernel,
                hf_weight_key=f"{prefix}.self_attn.v_proj.weight",
                hook_fn=transpose_and_reshape,
            )
            # Output projection.
            loader.port_weight(
                keras_variable=attn._output_dense.kernel,
                hf_weight_key=f"{prefix}.self_attn.o_proj.weight",
                hook_fn=transpose_and_reshape,
            )

        elif layer_type == "linear_attention":
            gdn = decoder_layer._linear_attn

            # QKV fused projection.
            loader.port_weight(
                keras_variable=gdn.in_proj_qkv.kernel,
                hf_weight_key=f"{prefix}.linear_attn.in_proj_qkv.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )
            # Z (output gate) projection.
            loader.port_weight(
                keras_variable=gdn.in_proj_z.kernel,
                hf_weight_key=f"{prefix}.linear_attn.in_proj_z.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )
            # B (write gate) projection.
            loader.port_weight(
                keras_variable=gdn.in_proj_b.kernel,
                hf_weight_key=f"{prefix}.linear_attn.in_proj_b.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )
            # A (decay gate) projection.
            loader.port_weight(
                keras_variable=gdn.in_proj_a.kernel,
                hf_weight_key=f"{prefix}.linear_attn.in_proj_a.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )
            # Conv1d weight: HF shape (channels, 1, kernel_size) ->
            # KerasHub shape (channels, kernel_size).
            loader.port_weight(
                keras_variable=gdn.conv1d_weight,
                hf_weight_key=f"{prefix}.linear_attn.conv1d.weight",
                hook_fn=lambda hf_tensor, _: np.squeeze(hf_tensor, axis=1),
            )
            # dt_bias.
            loader.port_weight(
                keras_variable=gdn.dt_bias,
                hf_weight_key=f"{prefix}.linear_attn.dt_bias",
            )
            # A_log.
            loader.port_weight(
                keras_variable=gdn.A_log,
                hf_weight_key=f"{prefix}.linear_attn.A_log",
            )
            # Output gated RMSNorm.
            loader.port_weight(
                keras_variable=gdn.norm.scale,
                hf_weight_key=f"{prefix}.linear_attn.norm.weight",
                hook_fn=lambda hf_tensor, _: hf_tensor - 1.0,
            )
            # Output projection.
            loader.port_weight(
                keras_variable=gdn.out_proj.kernel,
                hf_weight_key=f"{prefix}.linear_attn.out_proj.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )

        # MLP layers (same for both layer types).
        loader.port_weight(
            keras_variable=(
                decoder_layer._feedforward_intermediate_dense.kernel
            ),
            hf_weight_key=f"{prefix}.mlp.up_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=(decoder_layer._feedforward_output_dense.kernel),
            hf_weight_key=f"{prefix}.mlp.down_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=(decoder_layer._feedforward_gate_dense.kernel),
            hf_weight_key=f"{prefix}.mlp.gate_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )

        # Post-attention layernorm.
        loader.port_weight(
            keras_variable=(decoder_layer._post_attention_layernorm.scale),
            hf_weight_key=f"{prefix}.post_attention_layernorm.weight",
        )

    # Final normalization layer.
    loader.port_weight(
        keras_variable=backbone.get_layer("sequence_output_layernorm").scale,
        hf_weight_key="model.norm.weight",
    )

    # === Vision encoder weights (optional) ===
    if (
        hasattr(backbone, "vision_encoder")
        and backbone.vision_encoder is not None
    ):
        vis = backbone.vision_encoder
        vis_prefix = "model.visual"

        # Patch embedding Conv3D.
        # HF: (hidden_size, in_channels, temporal, patch, patch)
        # Keras Conv3D: (temporal, patch, patch, in_channels, hidden_size)
        loader.port_weight(
            keras_variable=vis.patch_embed.proj.kernel,
            hf_weight_key=f"{vis_prefix}.patch_embed.proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor, (2, 3, 4, 1, 0)
            ),
        )
        loader.port_weight(
            keras_variable=vis.patch_embed.proj.bias,
            hf_weight_key=f"{vis_prefix}.patch_embed.proj.bias",
        )

        # Absolute position embedding.
        loader.port_weight(
            keras_variable=vis.pos_embed.embeddings,
            hf_weight_key=f"{vis_prefix}.pos_embed.weight",
        )

        # ViT blocks.
        for blk_i in range(vis.depth):
            blk = vis.blocks[blk_i]
            blk_prefix = f"{vis_prefix}.blocks.{blk_i}"

            # LayerNorm 1 & 2.
            loader.port_weight(
                keras_variable=blk.norm1.gamma,
                hf_weight_key=f"{blk_prefix}.norm1.weight",
            )
            loader.port_weight(
                keras_variable=blk.norm1.beta,
                hf_weight_key=f"{blk_prefix}.norm1.bias",
            )
            loader.port_weight(
                keras_variable=blk.norm2.gamma,
                hf_weight_key=f"{blk_prefix}.norm2.weight",
            )
            loader.port_weight(
                keras_variable=blk.norm2.beta,
                hf_weight_key=f"{blk_prefix}.norm2.bias",
            )

            # Attention QKV (fused) and output projection.
            loader.port_weight(
                keras_variable=blk.attn.qkv.kernel,
                hf_weight_key=f"{blk_prefix}.attn.qkv.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, (1, 0)),
            )
            loader.port_weight(
                keras_variable=blk.attn.qkv.bias,
                hf_weight_key=f"{blk_prefix}.attn.qkv.bias",
            )
            loader.port_weight(
                keras_variable=blk.attn.proj.kernel,
                hf_weight_key=f"{blk_prefix}.attn.proj.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, (1, 0)),
            )
            loader.port_weight(
                keras_variable=blk.attn.proj.bias,
                hf_weight_key=f"{blk_prefix}.attn.proj.bias",
            )

            # MLP fc1 and fc2.
            loader.port_weight(
                keras_variable=blk.mlp.fc1.kernel,
                hf_weight_key=f"{blk_prefix}.mlp.fc1.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, (1, 0)),
            )
            loader.port_weight(
                keras_variable=blk.mlp.fc1.bias,
                hf_weight_key=f"{blk_prefix}.mlp.fc1.bias",
            )
            loader.port_weight(
                keras_variable=blk.mlp.fc2.kernel,
                hf_weight_key=f"{blk_prefix}.mlp.fc2.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, (1, 0)),
            )
            loader.port_weight(
                keras_variable=blk.mlp.fc2.bias,
                hf_weight_key=f"{blk_prefix}.mlp.fc2.bias",
            )

        # Patch merger.
        merger_prefix = f"{vis_prefix}.merger"
        loader.port_weight(
            keras_variable=vis.merger.norm.gamma,
            hf_weight_key=f"{merger_prefix}.ln_q.weight",
        )
        loader.port_weight(
            keras_variable=vis.merger.norm.beta,
            hf_weight_key=f"{merger_prefix}.ln_q.bias",
        )
        # mlp.0 = fc1, mlp.2 = fc2
        loader.port_weight(
            keras_variable=vis.merger.fc1.kernel,
            hf_weight_key=f"{merger_prefix}.mlp.0.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, (1, 0)),
        )
        loader.port_weight(
            keras_variable=vis.merger.fc1.bias,
            hf_weight_key=f"{merger_prefix}.mlp.0.bias",
        )
        loader.port_weight(
            keras_variable=vis.merger.fc2.kernel,
            hf_weight_key=f"{merger_prefix}.mlp.2.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, (1, 0)),
        )
        loader.port_weight(
            keras_variable=vis.merger.fc2.bias,
            hf_weight_key=f"{merger_prefix}.mlp.2.bias",
        )

    return backbone


def convert_tokenizer(cls, preset, **kwargs):
    tokenizer_config = load_json(preset, "tokenizer.json")
    vocab = tokenizer_config["model"]["vocab"]
    merges = tokenizer_config["model"]["merges"]
    # Merges may be lists (["Ġ", "a"]) or already strings ("Ġ a").
    if merges and isinstance(merges[0], list):
        merges = [" ".join(item) for item in merges]

    # Load all special tokens except "reserved" ones.
    special_tokens = set()
    for token in tokenizer_config["added_tokens"]:
        if not token["content"].startswith("<|reserved_special_token_"):
            vocab[token["content"]] = token["id"]
            special_tokens.add(token["content"])

    kwargs.update(
        {
            "unsplittable_tokens": list(special_tokens),
        }
    )

    return cls(vocabulary=vocab, merges=merges, **kwargs)
