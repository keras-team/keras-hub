"""HF -> KerasHub weight converter for Qwen3.5 MoE."""

import numpy as np

from keras_hub.src.models.qwen3_5.qwen3_5_vision_encoder import (
    Qwen3_5VisionEncoder,
)
from keras_hub.src.models.qwen3_5_moe.qwen3_5_moe_backbone import (
    Qwen3_5MoeBackbone,
)
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = Qwen3_5MoeBackbone


def _transpose_and_reshape(x, shape):
    """Transpose a 2-D HF weight and reshape to the target Keras shape."""
    return np.reshape(np.transpose(x), shape)


def _compute_scale_offset(image_mean, image_std, rescale_factor=1.0 / 255):
    """Compute KH scale/offset from HF image_mean and image_std.

    KH applies:  output = input * scale + offset
    HF applies:
        1. Rescale: x = x * rescale_factor   (e.g. 1/255)
        2. Normalize: x = (x - mean) / std
    Combined:  scale = rescale_factor / std,  offset = -mean / std
    """
    scale = [rescale_factor / s for s in image_std]
    offset = [-m / s for m, s in zip(image_mean, image_std)]
    return scale, offset


def load_image_converter_config(preset, transformers_config):
    """Return kwargs for Qwen3_5ImageConverter, or None for text-only."""
    if "vision_config" not in transformers_config:
        return None

    vision_config = transformers_config["vision_config"]
    preprocessor_config = load_json(preset, "preprocessor_config.json")

    scale, offset = _compute_scale_offset(
        preprocessor_config["image_mean"],
        preprocessor_config["image_std"],
    )

    size_config = preprocessor_config["size"]
    return {
        "patch_size": vision_config["patch_size"],
        "temporal_patch_size": vision_config["temporal_patch_size"],
        "spatial_merge_size": vision_config["spatial_merge_size"],
        "min_pixels": size_config["shortest_edge"],
        "max_pixels": size_config["longest_edge"],
        "scale": scale,
        "offset": offset,
        "interpolation": "bicubic",
        "antialias": True,
    }


def load_video_converter_config(preset, transformers_config):
    """Return kwargs for Qwen3_5VideoConverter, or None for text-only.

    Loads ``video_preprocessor_config.json`` which has video-specific
    pixel budgets that differ from the image preprocessor config.
    """
    if "vision_config" not in transformers_config:
        return None

    vision_config = transformers_config["vision_config"]
    video_config = load_json(preset, "video_preprocessor_config.json")

    scale, offset = _compute_scale_offset(
        video_config["image_mean"],
        video_config["image_std"],
    )

    size_config = video_config["size"]
    return {
        "patch_size": video_config.get(
            "patch_size", vision_config["patch_size"]
        ),
        "temporal_patch_size": video_config.get(
            "temporal_patch_size",
            vision_config["temporal_patch_size"],
        ),
        "spatial_merge_size": video_config.get(
            "merge_size", vision_config["spatial_merge_size"]
        ),
        "min_pixels": size_config["shortest_edge"],
        "max_pixels": size_config["longest_edge"],
        "scale": scale,
        "offset": offset,
        "interpolation": "bicubic",
        "antialias": True,
    }


def convert_backbone_config(transformers_config):
    tie_word_embeddings = transformers_config["tie_word_embeddings"]

    top_level_vision_config = transformers_config.get("vision_config", None)
    top_level_hidden_size = transformers_config.get("hidden_size", None)

    if "text_config" in transformers_config:
        transformers_config = transformers_config["text_config"]

    rope_params = transformers_config["rope_parameters"]

    mrope_section = rope_params.get("mrope_section", None)

    num_layers = transformers_config["num_hidden_layers"]
    layer_types = transformers_config.get("layer_types", None)
    if layer_types is None:
        layer_types = [
            ("linear_attention" if bool((i + 1) % 4) else "full_attention")
            for i in range(num_layers)
        ]

    vision_encoder = None
    vision_config = top_level_vision_config
    if vision_config is not None:
        text_hidden = transformers_config.get(
            "hidden_size", top_level_hidden_size
        )
        vision_encoder = Qwen3_5VisionEncoder(
            depth=vision_config["depth"],
            hidden_size=vision_config["hidden_size"],
            num_heads=vision_config["num_heads"],
            intermediate_size=vision_config["intermediate_size"],
            in_channels=vision_config["in_channels"],
            patch_size=vision_config["patch_size"],
            temporal_patch_size=vision_config["temporal_patch_size"],
            spatial_merge_size=vision_config["spatial_merge_size"],
            out_hidden_size=text_hidden or vision_config["out_hidden_size"],
            num_position_embeddings=vision_config["num_position_embeddings"],
        )

    result = {
        "vocabulary_size": transformers_config["vocab_size"],
        "head_dim": transformers_config["head_dim"],
        "hidden_dim": transformers_config["hidden_size"],
        "num_layers": num_layers,
        "num_query_heads": transformers_config["num_attention_heads"],
        "num_key_value_heads": transformers_config["num_key_value_heads"],
        "moe_intermediate_dim": transformers_config["moe_intermediate_size"],
        "shared_expert_intermediate_size": transformers_config[
            "shared_expert_intermediate_size"
        ],
        "num_experts": transformers_config["num_experts"],
        "top_k": transformers_config["num_experts_per_tok"],
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
        "router_aux_loss_coefficient": transformers_config.get(
            "router_aux_loss_coef", 0.001
        ),
    }
    if vision_encoder is not None:
        result["vision_encoder"] = vision_encoder
    if mrope_section is not None:
        result["mrope_section"] = mrope_section
    return result


def convert_weights(backbone, loader, transformers_config):
    ported_hf_keys = set()

    def _port(keras_var, hf_key, hook_fn=None):
        """Port a single weight and track the HF key."""
        loader.port_weight(keras_var, hf_key, hook_fn=hook_fn)
        ported_hf_keys.add(hf_key)

    # Text model weights.

    _port(
        backbone.get_layer("token_embedding").embeddings,
        "model.language_model.embed_tokens.weight",
    )
    if not backbone.tie_word_embeddings:
        _port(
            backbone.get_layer("token_embedding").reverse_embeddings,
            "lm_head.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"transformer_layer_{i}")
        layer_type = decoder_layer.layer_type
        prefix = f"model.language_model.layers.{i}"

        _port(
            decoder_layer._input_layernorm.scale,
            f"{prefix}.input_layernorm.weight",
        )

        if layer_type == "full_attention":
            attn = decoder_layer._self_attention_layer

            _port(
                attn._query_dense.kernel,
                f"{prefix}.self_attn.q_proj.weight",
                hook_fn=_transpose_and_reshape,
            )
            _port(
                attn._query_norm.scale,
                f"{prefix}.self_attn.q_norm.weight",
            )
            _port(
                attn._key_dense.kernel,
                f"{prefix}.self_attn.k_proj.weight",
                hook_fn=_transpose_and_reshape,
            )
            _port(
                attn._key_norm.scale,
                f"{prefix}.self_attn.k_norm.weight",
            )
            _port(
                attn._value_dense.kernel,
                f"{prefix}.self_attn.v_proj.weight",
                hook_fn=_transpose_and_reshape,
            )
            _port(
                attn._output_dense.kernel,
                f"{prefix}.self_attn.o_proj.weight",
                hook_fn=_transpose_and_reshape,
            )

        elif layer_type == "linear_attention":
            gdn = decoder_layer._linear_attn

            _port(
                gdn.in_proj_qkv.kernel,
                f"{prefix}.linear_attn.in_proj_qkv.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )
            _port(
                gdn.in_proj_z.kernel,
                f"{prefix}.linear_attn.in_proj_z.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )
            _port(
                gdn.in_proj_b.kernel,
                f"{prefix}.linear_attn.in_proj_b.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )
            _port(
                gdn.in_proj_a.kernel,
                f"{prefix}.linear_attn.in_proj_a.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )
            # Conv1d: HF (ch, 1, kernel) -> Keras (ch, kernel).
            _port(
                gdn.conv1d_weight,
                f"{prefix}.linear_attn.conv1d.weight",
                hook_fn=lambda hf_tensor, _: np.squeeze(hf_tensor, axis=1),
            )
            _port(gdn.dt_bias, f"{prefix}.linear_attn.dt_bias")
            _port(gdn.A_log, f"{prefix}.linear_attn.A_log")
            _port(
                gdn.norm.scale,
                f"{prefix}.linear_attn.norm.weight",
            )
            _port(
                gdn.out_proj.kernel,
                f"{prefix}.linear_attn.out_proj.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )

        moe_block = decoder_layer.mlp

        _port(
            moe_block._router_gate.kernel,
            f"{prefix}.mlp.gate.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )

        _port(
            moe_block.shared_expert._feedforward_gate_dense.kernel,
            f"{prefix}.mlp.shared_expert.gate_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        _port(
            moe_block.shared_expert._feedforward_intermediate_dense.kernel,
            f"{prefix}.mlp.shared_expert.up_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        _port(
            moe_block.shared_expert._feedforward_output_dense.kernel,
            f"{prefix}.mlp.shared_expert.down_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )

        _port(
            moe_block._shared_expert_gate.kernel,
            f"{prefix}.mlp.shared_expert_gate.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )

        num_experts = backbone.num_experts
        try:
            gate_up = loader.get_tensor(f"{prefix}.mlp.experts.gate_up_proj")
            down = loader.get_tensor(f"{prefix}.mlp.experts.down_proj")
            ported_hf_keys.add(f"{prefix}.mlp.experts.gate_up_proj")
            ported_hf_keys.add(f"{prefix}.mlp.experts.down_proj")

            gate_up = np.transpose(gate_up, axes=(0, 2, 1))
            down = np.transpose(down, axes=(0, 2, 1))

            moe_block.expert_bank._expert_feedforward_gate_dense.assign(gate_up)
            moe_block.expert_bank._expert_feedforward_output_dense.assign(down)
        except Exception:
            gate_up_proj_list = []
            down_proj_list = []
            for expert_idx in range(num_experts):
                gate_proj = loader.get_tensor(
                    f"{prefix}.mlp.experts.{expert_idx}.gate_proj.weight"
                )
                up_proj = loader.get_tensor(
                    f"{prefix}.mlp.experts.{expert_idx}.up_proj.weight"
                )
                gate_proj = np.transpose(gate_proj, axes=(1, 0))
                up_proj = np.transpose(up_proj, axes=(1, 0))
                gate_up_proj = np.concatenate([gate_proj, up_proj], axis=-1)
                gate_up_proj_list.append(gate_up_proj)

                down_proj = loader.get_tensor(
                    f"{prefix}.mlp.experts.{expert_idx}.down_proj.weight"
                )
                down_proj = np.transpose(down_proj, axes=(1, 0))
                down_proj_list.append(down_proj)

                ported_hf_keys.add(
                    f"{prefix}.mlp.experts.{expert_idx}.gate_proj.weight"
                )
                ported_hf_keys.add(
                    f"{prefix}.mlp.experts.{expert_idx}.up_proj.weight"
                )
                ported_hf_keys.add(
                    f"{prefix}.mlp.experts.{expert_idx}.down_proj.weight"
                )

            gate_up_proj_batched = np.stack(gate_up_proj_list, axis=0)
            down_proj_batched = np.stack(down_proj_list, axis=0)

            moe_block.expert_bank._expert_feedforward_gate_dense.assign(
                gate_up_proj_batched
            )
            moe_block.expert_bank._expert_feedforward_output_dense.assign(
                down_proj_batched
            )

        _port(
            decoder_layer._post_attention_layernorm.scale,
            f"{prefix}.post_attention_layernorm.weight",
        )

    _port(
        backbone.get_layer("sequence_output_layernorm").scale,
        "model.language_model.norm.weight",
    )

    # Vision encoder weights (optional).
    if (
        hasattr(backbone, "vision_encoder")
        and backbone.vision_encoder is not None
    ):
        vis = backbone.vision_encoder
        vis_prefix = "model.visual"

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

        _port(
            vis.patch_embed.proj.kernel,
            f"{vis_prefix}.patch_embed.proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor, (2, 3, 4, 1, 0)
            ),
        )
        _port(
            vis.patch_embed.proj.bias,
            f"{vis_prefix}.patch_embed.proj.bias",
        )

        _port(
            vis.pos_embed.embeddings,
            f"{vis_prefix}.pos_embed.weight",
        )

        for blk_i in range(vis.depth):
            blk = vis.blocks[blk_i]
            blk_prefix = f"{vis_prefix}.blocks.{blk_i}"

            _port(blk.norm1.gamma, f"{blk_prefix}.norm1.weight")
            _port(blk.norm1.beta, f"{blk_prefix}.norm1.bias")
            _port(blk.norm2.gamma, f"{blk_prefix}.norm2.weight")
            _port(blk.norm2.beta, f"{blk_prefix}.norm2.bias")

            _port(
                blk.attn.qkv.kernel,
                f"{blk_prefix}.attn.qkv.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, (1, 0)),
            )
            _port(blk.attn.qkv.bias, f"{blk_prefix}.attn.qkv.bias")
            _port(
                blk.attn.proj.kernel,
                f"{blk_prefix}.attn.proj.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, (1, 0)),
            )
            _port(blk.attn.proj.bias, f"{blk_prefix}.attn.proj.bias")

            _port(
                blk.mlp.fc1.kernel,
                f"{blk_prefix}.mlp.linear_fc1.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, (1, 0)),
            )
            _port(blk.mlp.fc1.bias, f"{blk_prefix}.mlp.linear_fc1.bias")
            _port(
                blk.mlp.fc2.kernel,
                f"{blk_prefix}.mlp.linear_fc2.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, (1, 0)),
            )
            _port(blk.mlp.fc2.bias, f"{blk_prefix}.mlp.linear_fc2.bias")

        merger_prefix = f"{vis_prefix}.merger"
        _port(vis.merger.norm.gamma, f"{merger_prefix}.norm.weight")
        _port(vis.merger.norm.beta, f"{merger_prefix}.norm.bias")
        _port(
            vis.merger.fc1.kernel,
            f"{merger_prefix}.linear_fc1.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, (1, 0)),
        )
        _port(vis.merger.fc1.bias, f"{merger_prefix}.linear_fc1.bias")
        _port(
            vis.merger.fc2.kernel,
            f"{merger_prefix}.linear_fc2.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, (1, 0)),
        )
        _port(vis.merger.fc2.bias, f"{merger_prefix}.linear_fc2.bias")

    # Dynamic weight audit.
    _audit_weights(backbone, loader, ported_hf_keys)

    return backbone


def _audit_weights(backbone, loader, ported_hf_keys):
    """Audit that all relevant HF weights were ported."""
    if loader.safetensor_config is None:
        print(
            "  [AUDIT] No safetensor config found - "
            "skipping dynamic weight audit."
        )
        return

    all_hf_keys = set(loader.safetensor_config["weight_map"].keys())

    skip_prefixes = ("mtp.",)

    skipped_keys = set()
    for key in all_hf_keys:
        if any(key.startswith(p) for p in skip_prefixes):
            skipped_keys.add(key)

    expected_keys = all_hf_keys - skipped_keys

    normalized_ported = set()
    for key in ported_hf_keys:
        normalized_ported.add(key)
        if loader.prefix and not key.startswith(loader.prefix):
            normalized_ported.add(loader.prefix + key)

    unported = expected_keys - normalized_ported
    if unported:
        print(f"\n  [AUDIT] WARNING: {len(unported)} HF weight(s) NOT ported:")
        for key in sorted(unported):
            print(f"    - {key}")
    else:
        print(
            f"\n  [AUDIT] All {len(expected_keys)} expected HF weights "
            f"ported successfully! ({len(skipped_keys)} mtp keys skipped)"
        )

    keras_var_count = backbone.count_params()
    print(f"  [AUDIT] Total KerasHub parameters: {keras_var_count:,}")


def convert_tokenizer(cls, preset, **kwargs):
    tokenizer_config = load_json(preset, "tokenizer.json")
    vocab = tokenizer_config["model"]["vocab"]
    merges = tokenizer_config["model"]["merges"]
    if merges and isinstance(merges[0], list):
        merges = [" ".join(item) for item in merges]

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
