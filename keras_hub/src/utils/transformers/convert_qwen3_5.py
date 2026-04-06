"""HF -> KerasHub weight converter for Qwen3.5."""

import numpy as np

from keras_hub.src.models.qwen3_5.qwen3_5_backbone import Qwen3_5Backbone
from keras_hub.src.models.qwen3_5.qwen3_5_vision_encoder import (
    Qwen3_5VisionEncoder,
)
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = Qwen3_5Backbone


def _transpose_and_reshape(x, shape):
    """Transpose a 2-D HF weight and reshape to the target Keras shape."""
    return np.reshape(np.transpose(x), shape)


def load_image_converter_config(preset, transformers_config):
    """Return kwargs for Qwen3_5ImageConverter, or None for text-only."""
    if "vision_config" not in transformers_config:
        return None

    vision_config = transformers_config["vision_config"]
    preprocessor_config = load_json(preset, "preprocessor_config.json")

    return {
        "patch_size": vision_config.get("patch_size", 16),
        "temporal_patch_size": vision_config.get("temporal_patch_size", 2),
        "spatial_merge_size": vision_config.get("spatial_merge_size", 2),
        "min_pixels": preprocessor_config.get("min_pixels", 65536),
        "max_pixels": preprocessor_config.get("max_pixels", 16777216),
        "image_mean": preprocessor_config.get("image_mean", [0.5, 0.5, 0.5]),
        "image_std": preprocessor_config.get("image_std", [0.5, 0.5, 0.5]),
    }


def convert_backbone_config(transformers_config):
    # tie_word_embeddings is at the top-level config.
    tie_word_embeddings = transformers_config["tie_word_embeddings"]

    # Save top-level fields that live outside text_config.
    top_level_vision_config = transformers_config.get("vision_config", None)
    top_level_hidden_size = transformers_config.get("hidden_size", None)

    # Qwen3.5 text config is nested under "text_config".
    if "text_config" in transformers_config:
        transformers_config = transformers_config["text_config"]

    # rope_theta and partial_rotary_factor are nested under
    # rope_parameters in the HF config.
    rope_params = transformers_config["rope_parameters"]

    # M-RoPE section lives inside rope_parameters in HF config.
    mrope_section = rope_params.get("mrope_section", None)

    # Build layer_types list.
    num_layers = transformers_config["num_hidden_layers"]
    layer_types = transformers_config.get("layer_types", None)
    if layer_types is None:
        # Default: every 4th layer is full_attention.
        layer_types = [
            ("linear_attention" if bool((i + 1) % 4) else "full_attention")
            for i in range(num_layers)
        ]

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
    # ----------------------------------------------------------------
    # Track which HF keys we actually port so we can audit at the end.
    # ----------------------------------------------------------------
    ported_hf_keys = set()

    def _port(keras_var, hf_key, hook_fn=None):
        """Port a single weight and track the HF key."""
        loader.port_weight(keras_var, hf_key, hook_fn=hook_fn)
        ported_hf_keys.add(hf_key)

    # ------------------------------------------------------------------
    # Text model weights
    # ------------------------------------------------------------------
    # The HF weight prefix is "model.language_model." but the
    # SafetensorLoader.get_prefixed_key() auto-discovers the prefix on
    # the first lookup, so we use short keys like "model.layers.{i}".
    # For Qwen3.5 the safetensors keys start with
    # "model.language_model." — the loader handles this transparently.
    # ------------------------------------------------------------------

    # Embedding.
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

        # Input layernorm.
        _port(
            decoder_layer._input_layernorm.scale,
            f"{prefix}.input_layernorm.weight",
        )

        if layer_type == "full_attention":
            attn = decoder_layer._self_attention_layer

            # Q projection (includes gate: head_dim * 2).
            _port(
                attn._query_dense.kernel,
                f"{prefix}.self_attn.q_proj.weight",
                hook_fn=_transpose_and_reshape,
            )
            # Q norm.
            _port(
                attn._query_norm.scale,
                f"{prefix}.self_attn.q_norm.weight",
            )
            # K projection.
            _port(
                attn._key_dense.kernel,
                f"{prefix}.self_attn.k_proj.weight",
                hook_fn=_transpose_and_reshape,
            )
            # K norm.
            _port(
                attn._key_norm.scale,
                f"{prefix}.self_attn.k_norm.weight",
            )
            # V projection.
            _port(
                attn._value_dense.kernel,
                f"{prefix}.self_attn.v_proj.weight",
                hook_fn=_transpose_and_reshape,
            )
            # Output projection.
            _port(
                attn._output_dense.kernel,
                f"{prefix}.self_attn.o_proj.weight",
                hook_fn=_transpose_and_reshape,
            )

        elif layer_type == "linear_attention":
            gdn = decoder_layer._linear_attn

            # QKV fused projection.
            _port(
                gdn.in_proj_qkv.kernel,
                f"{prefix}.linear_attn.in_proj_qkv.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )
            # Z (output gate) projection.
            _port(
                gdn.in_proj_z.kernel,
                f"{prefix}.linear_attn.in_proj_z.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )
            # B (write gate) projection.
            _port(
                gdn.in_proj_b.kernel,
                f"{prefix}.linear_attn.in_proj_b.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )
            # A (decay gate) projection.
            _port(
                gdn.in_proj_a.kernel,
                f"{prefix}.linear_attn.in_proj_a.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )
            # Conv1d weight: HF shape (channels, 1, kernel_size) ->
            # KerasHub shape (channels, kernel_size).
            _port(
                gdn.conv1d_weight,
                f"{prefix}.linear_attn.conv1d.weight",
                hook_fn=lambda hf_tensor, _: np.squeeze(hf_tensor, axis=1),
            )
            # dt_bias.
            _port(
                gdn.dt_bias,
                f"{prefix}.linear_attn.dt_bias",
            )
            # A_log.
            _port(
                gdn.A_log,
                f"{prefix}.linear_attn.A_log",
            )
            # Output gated RMSNorm (uses direct weight * x, not (1+w)*x).
            _port(
                gdn.norm.scale,
                f"{prefix}.linear_attn.norm.weight",
            )
            # Output projection.
            _port(
                gdn.out_proj.kernel,
                f"{prefix}.linear_attn.out_proj.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )

        # MLP layers (same for both layer types).
        _port(
            decoder_layer._feedforward_intermediate_dense.kernel,
            f"{prefix}.mlp.up_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        _port(
            decoder_layer._feedforward_output_dense.kernel,
            f"{prefix}.mlp.down_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        _port(
            decoder_layer._feedforward_gate_dense.kernel,
            f"{prefix}.mlp.gate_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )

        # Post-attention layernorm.
        _port(
            decoder_layer._post_attention_layernorm.scale,
            f"{prefix}.post_attention_layernorm.weight",
        )

    # Final normalization layer.
    _port(
        backbone.get_layer("sequence_output_layernorm").scale,
        "model.language_model.norm.weight",
    )

    # ------------------------------------------------------------------
    # Vision encoder weights (optional)
    # ------------------------------------------------------------------
    if (
        hasattr(backbone, "vision_encoder")
        and backbone.vision_encoder is not None
    ):
        vis = backbone.vision_encoder
        vis_prefix = "model.visual"

        # Explicitly build sublayers since they use lazy build().
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

        # Patch embedding Conv3D.
        # HF: (hidden_size, in_channels, temporal, patch, patch)
        # Keras Conv3D: (temporal, patch, patch, in_channels, hidden_size)
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

        # Absolute position embedding.
        _port(
            vis.pos_embed.embeddings,
            f"{vis_prefix}.pos_embed.weight",
        )

        # ViT blocks.
        for blk_i in range(vis.depth):
            blk = vis.blocks[blk_i]
            blk_prefix = f"{vis_prefix}.blocks.{blk_i}"

            # LayerNorm 1 & 2.
            _port(
                blk.norm1.gamma,
                f"{blk_prefix}.norm1.weight",
            )
            _port(
                blk.norm1.beta,
                f"{blk_prefix}.norm1.bias",
            )
            _port(
                blk.norm2.gamma,
                f"{blk_prefix}.norm2.weight",
            )
            _port(
                blk.norm2.beta,
                f"{blk_prefix}.norm2.bias",
            )

            # Attention QKV (fused) and output projection.
            _port(
                blk.attn.qkv.kernel,
                f"{blk_prefix}.attn.qkv.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, (1, 0)),
            )
            _port(
                blk.attn.qkv.bias,
                f"{blk_prefix}.attn.qkv.bias",
            )
            _port(
                blk.attn.proj.kernel,
                f"{blk_prefix}.attn.proj.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, (1, 0)),
            )
            _port(
                blk.attn.proj.bias,
                f"{blk_prefix}.attn.proj.bias",
            )

            # MLP fc1 and fc2.
            _port(
                blk.mlp.fc1.kernel,
                f"{blk_prefix}.mlp.linear_fc1.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, (1, 0)),
            )
            _port(
                blk.mlp.fc1.bias,
                f"{blk_prefix}.mlp.linear_fc1.bias",
            )
            _port(
                blk.mlp.fc2.kernel,
                f"{blk_prefix}.mlp.linear_fc2.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, (1, 0)),
            )
            _port(
                blk.mlp.fc2.bias,
                f"{blk_prefix}.mlp.linear_fc2.bias",
            )

        # Patch merger.
        merger_prefix = f"{vis_prefix}.merger"
        _port(
            vis.merger.norm.gamma,
            f"{merger_prefix}.norm.weight",
        )
        _port(
            vis.merger.norm.beta,
            f"{merger_prefix}.norm.bias",
        )
        # mlp.0 = fc1, mlp.2 = fc2 -> linear_fc1, linear_fc2 (HF Qwen3.5)
        _port(
            vis.merger.fc1.kernel,
            f"{merger_prefix}.linear_fc1.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, (1, 0)),
        )
        _port(
            vis.merger.fc1.bias,
            f"{merger_prefix}.linear_fc1.bias",
        )
        _port(
            vis.merger.fc2.kernel,
            f"{merger_prefix}.linear_fc2.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, (1, 0)),
        )
        _port(
            vis.merger.fc2.bias,
            f"{merger_prefix}.linear_fc2.bias",
        )

    # ----------------------------------------------------------------
    # Dynamic weight audit
    # ----------------------------------------------------------------
    _audit_weights(backbone, loader, ported_hf_keys)

    return backbone


def _audit_weights(backbone, loader, ported_hf_keys):
    """Audit that all relevant HF weights were ported.

    Compares the set of ported HF keys against the full weight map from
    the safetensors config. Reports any HF keys that were NOT ported
    (excluding known skip patterns like `mtp.*` multi-token prediction).
    """
    # Get all HF weight keys from the safetensors config.
    if loader.safetensor_config is None:
        print(
            "  [AUDIT] No safetensor config found — "
            "skipping dynamic weight audit."
        )
        return

    all_hf_keys = set(loader.safetensor_config["weight_map"].keys())

    # Keys to intentionally skip (not part of standard inference).
    skip_prefixes = ("mtp.",)

    skipped_keys = set()
    for key in all_hf_keys:
        if any(key.startswith(p) for p in skip_prefixes):
            skipped_keys.add(key)

    expected_keys = all_hf_keys - skipped_keys

    # Normalize ported keys: the loader may have added a prefix.
    # We need to check both the raw key and the prefixed key.
    normalized_ported = set()
    for key in ported_hf_keys:
        normalized_ported.add(key)
        # Also add the prefixed version if loader discovered a prefix.
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

    # Also count Keras variables for reference.
    keras_var_count = backbone.count_params()
    print(f"  [AUDIT] Total KerasHub parameters: {keras_var_count:,}")


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
