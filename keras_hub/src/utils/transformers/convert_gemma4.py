import re

import numpy as np

from keras_hub.src.models.gemma4.gemma4_audio_encoder import Gemma4AudioEncoder
from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone
from keras_hub.src.models.gemma4.gemma4_tokenizer import IMAGE_PLACEHOLDER_TOKEN
from keras_hub.src.models.gemma4.gemma4_vision_encoder import (
    Gemma4VisionEncoder,
)
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = Gemma4Backbone


def load_image_converter_config(preset, transformers_config):
    """Return kwargs for Gemma4ImageConverter, or None for text-only models."""
    if "vision_config" not in transformers_config:
        return None

    processor_config = load_json(preset, "processor_config.json")
    image_processor = processor_config.get("image_processor", processor_config)
    mean = image_processor["image_mean"]
    std = image_processor["image_std"]
    rescale_factor = image_processor.get("rescale_factor", 1.0 / 255.0)

    offset = [(-m / s) for m, s in zip(mean, std)]
    scale = [(rescale_factor / s) for s in std]

    vision_config = transformers_config["vision_config"]
    image_size = vision_config.get("image_size", 896)
    patch_size = vision_config.get("patch_size", 14)

    return {
        "image_size": (image_size, image_size),
        "patch_size": patch_size,
        "scale": scale,
        "offset": offset,
    }


def load_audio_converter_config(preset, transformers_config):
    """Return Gemma4AudioConverter kwargs, or None for text/vision models."""
    if "audio_config" not in transformers_config:
        return None
    return {}


def load_preprocessor_config(preset, transformers_config):
    """Return extra kwargs for Gemma4CausalLMPreprocessor from HF config.

    For audio-capable models (E2B/E4B), this wires `audio_input_feat_size`
    from the HF `audio_config` into the preprocessor so that it emits the
    correct audio dummy keys expected by the backbone.
    """
    config = {}
    if "vision_config" in transformers_config:
        vis_cfg = transformers_config["vision_config"]
        # Use the authoritative token count from the config rather than
        # recomputing it from image_size (which may not be present and
        # differs per-image for dynamic-resolution models).
        config["num_vision_tokens_per_image"] = vis_cfg["default_output_length"]

    audio_cfg = transformers_config.get("audio_config")
    if audio_cfg is not None:
        config["audio_input_feat_size"] = audio_cfg.get("input_feat_size", 128)
        config["num_audio_tokens_per_clip"] = audio_cfg.get(
            "num_audio_tokens_per_clip",
            750 // audio_cfg.get("reduction_factor", 1),
        )
    return config


def convert_backbone_config(transformers_config):
    """Map a Transformers config dict → Gemma4Backbone keyword arguments."""
    model_type = transformers_config.get("model_type", "gemma4")
    is_text_only = model_type == "gemma4_text"

    if is_text_only:
        text_cfg = transformers_config
        vision_encoder = None
        audio_encoder = None
        num_audio_tokens_per_clip = None
        image_size = None
    else:
        text_cfg = transformers_config["text_config"]
        image_size = 896

        # Vision encoder.
        if "vision_config" in transformers_config:
            vis_cfg = transformers_config["vision_config"]
            vision_encoder = Gemma4VisionEncoder(
                image_size=image_size,
                patch_size=vis_cfg["patch_size"],
                num_heads=vis_cfg["num_attention_heads"],
                hidden_dim=vis_cfg["hidden_size"],
                num_layers=vis_cfg["num_hidden_layers"],
                intermediate_dim=vis_cfg["intermediate_size"],
                head_dim=vis_cfg.get("head_dim", 64),
                num_key_value_heads=vis_cfg.get(
                    "num_key_value_heads", vis_cfg["num_attention_heads"]
                ),
                output_dim=text_cfg["hidden_size"],
                pool_size=vis_cfg.get("pooling_kernel_size", 3),
                position_embedding_size=vis_cfg.get(
                    "position_embedding_size", 10240
                ),
                rope_max_wavelength=vis_cfg.get("rope_parameters", {}).get(
                    "rope_theta", 100.0
                ),
                layer_norm_epsilon=vis_cfg.get("rms_norm_eps", 1e-6),
                use_clipped_linears=vis_cfg.get("use_clipped_linears", True),
                standardize=vis_cfg.get("standardize", False),
            )
        else:
            vision_encoder = None

        # Audio encoder — key may be present but null for vision-only models.
        if transformers_config.get("audio_config") is not None:
            aud_cfg = transformers_config["audio_config"]
            output_proj_dims = aud_cfg.get("output_proj_dims", 1536)
            audio_encoder = Gemma4AudioEncoder(
                input_feat_size=aud_cfg.get("input_feat_size", 128),
                hidden_size=aud_cfg.get("hidden_size", 1024),
                num_heads=aud_cfg.get("num_attention_heads", 8),
                num_layers=aud_cfg.get("num_hidden_layers", 12),
                chunk_size=aud_cfg.get("attention_chunk_size", 12),
                context_left=aud_cfg.get("attention_context_left", 13),
                context_right=aud_cfg.get("attention_context_right", 0),
                logit_cap=aud_cfg.get("attention_logit_cap", 50.0),
                invalid_logit_value=aud_cfg.get(
                    "attention_invalid_logits_value", -1e9
                ),
                conv_kernel_size=aud_cfg.get("conv_kernel_size", 5),
                reduction_factor=aud_cfg.get("reduction_factor", 1),
                residual_weight=aud_cfg.get("residual_weight", 0.5),
                gradient_clipping=aud_cfg.get("gradient_clipping", 1e10),
                sscp_conv_channels=tuple(
                    aud_cfg.get("subsampling_conv_channels", (128, 32))
                ),
                sscp_kernel_sizes=tuple(
                    tuple(k)
                    for k in aud_cfg.get(
                        "sscp_conv_kernel_size", ((3, 3), (3, 3))
                    )
                ),
                sscp_stride_sizes=tuple(
                    tuple(s)
                    for s in aud_cfg.get(
                        "sscp_conv_stride_size", ((2, 2), (2, 2))
                    )
                ),
                output_proj_dims=output_proj_dims,
                output_dim=text_cfg["hidden_size"],
                norm_eps=aud_cfg.get("rms_norm_eps", 1e-6),
                # HF uses rms_norm_eps for the SSCP LayerNorm too; there is
                # no separate sscp_conv_eps field in Gemma4AudioConfig.
                sscp_norm_eps=aud_cfg.get(
                    "sscp_conv_eps", aud_cfg.get("rms_norm_eps", 1e-6)
                ),
            )
            num_audio_tokens_per_clip = aud_cfg.get(
                "num_audio_tokens_per_clip",
                750 // aud_cfg.get("reduction_factor", 1),
            )
        else:
            audio_encoder = None
            num_audio_tokens_per_clip = None

    # Config stores the pattern as "_sliding_window_pattern" (underscore
    # prefix).
    if (
        "layer_types" in text_cfg
        and text_cfg["layer_types"]
        and len(text_cfg["layer_types"]) > 1
    ):
        layer_types = text_cfg["layer_types"]
        try:
            first_idx = layer_types.index("full_attention")
            second_idx = layer_types.index("full_attention", first_idx + 1)
            sliding_window_pattern = second_idx - first_idx
        except ValueError:
            sliding_window_pattern = 6
    else:
        sliding_window_pattern = (
            text_cfg.get("_sliding_window_pattern")
            or text_cfg.get("sliding_window_pattern")
            or 6
        )

    global_head_dim = text_cfg.get("global_head_dim", None)

    # Partial RoPE factor for global (full) attention layers.
    # Stored under rope_parameters["full_attention"]["partial_rotary_factor"].
    rope_params = text_cfg.get("rope_parameters", {}) or {}
    global_rope_partial_rotary_factor = rope_params.get(
        "full_attention", {}
    ).get("partial_rotary_factor")
    global_rope_theta = rope_params.get("full_attention", {}).get("rope_theta")
    local_rope_theta = rope_params.get("sliding_attention", {}).get(
        "rope_theta"
    )

    # If it's missing in `full_attention`, safely fall back to top-level cfg.
    if global_rope_theta is None:
        global_rope_theta = text_cfg.get("rope_theta")

    # HF `use_bidirectional_attention` controls vision-token attention only:
    #   null     → purely causal for all tokens (E2B, E4B).
    #   "vision" → vision tokens attend bidirectionally within an image
    #              while text tokens remain causal (26B, 31B and larger).
    # This maps to KH `use_vision_bidirectional_attention` only.
    # KH `use_bidirectional_attention` (full bidirec for ALL tokens) is a
    # separate concept used exclusively for embedding models and is never
    # set from this HF field.
    hf_bidir = text_cfg.get("use_bidirectional_attention")
    use_vision_bidirectional_attention = hf_bidir == "vision"

    return {
        "vocabulary_size": text_cfg.get("vocab_size", 262144),
        "image_size": image_size,
        "num_layers": text_cfg["num_hidden_layers"],
        "num_query_heads": text_cfg.get("num_attention_heads", 8),
        "num_key_value_heads": text_cfg.get("num_key_value_heads", 1),
        "hidden_dim": text_cfg["hidden_size"],
        "intermediate_dim": text_cfg["intermediate_size"],
        "head_dim": text_cfg["head_dim"],
        "global_head_dim": global_head_dim,
        "attention_logit_soft_cap": text_cfg.get(
            "attn_logit_softcapping", None
        ),
        "final_logit_soft_cap": text_cfg.get("final_logit_softcapping", None),
        "use_sliding_window_attention": text_cfg.get("sliding_window", 0) > 0,
        "sliding_window_size": text_cfg.get("sliding_window", 512) or 512,
        "sliding_window_pattern": sliding_window_pattern,
        "layer_norm_epsilon": text_cfg.get("rms_norm_eps", 1e-6),
        "vision_encoder": vision_encoder,
        "audio_encoder": audio_encoder,
        "num_audio_tokens_per_clip": num_audio_tokens_per_clip,
        "num_kv_shared_layers": text_cfg.get("num_kv_shared_layers", 0),
        "num_global_key_value_heads": text_cfg.get(
            "num_global_key_value_heads", None
        ),
        "hidden_size_per_layer_input": text_cfg.get(
            "hidden_size_per_layer_input"
        )
        or 0,
        "vocab_size_per_layer_input": text_cfg.get(
            "vocab_size_per_layer_input", None
        ),
        "global_rope_partial_rotary_factor": global_rope_partial_rotary_factor,
        "global_rope_wavelength": global_rope_theta,
        "local_rope_wavelength": local_rope_theta,
        "use_double_wide_mlp": text_cfg.get("use_double_wide_mlp", False),
        "enable_moe_block": text_cfg.get("enable_moe_block", False),
        "num_experts": text_cfg.get("num_experts", None),
        "expert_intermediate_dim": (
            text_cfg.get("moe_intermediate_size")
            or text_cfg.get("expert_intermediate_size")
        ),
        "num_experts_per_token": text_cfg.get("top_k_experts") or 8,
        "use_vision_bidirectional_attention": (
            use_vision_bidirectional_attention
        ),
    }


def convert_weights(backbone, loader, transformers_config):
    model_type = transformers_config.get("model_type", "gemma4")
    is_text_only = model_type == "gemma4_text"

    if is_text_only:
        text_prefix = _resolve_prefix(loader, ["model", "language_model", ""])
    else:
        text_prefix = _resolve_prefix(
            loader,
            ["model.language_model", "language_model"],
        )

    def hf_key(suffix):
        if text_prefix:
            return f"{text_prefix}.{suffix}"
        return suffix

    loader.port_weight(
        keras_variable=backbone.get_layer("token_embedding").embeddings,
        hf_weight_key=hf_key("embed_tokens.weight"),
    )

    # Per-layer token conditioning (E4B / E2B models).
    if backbone.hidden_size_per_layer_input > 0:
        loader.port_weight(
            keras_variable=backbone.get_layer(
                "per_layer_token_embedding"
            ).embeddings,
            hf_weight_key=hf_key("embed_tokens_per_layer.weight"),
        )
        loader.port_weight(
            keras_variable=backbone.get_layer(
                "per_layer_model_projection"
            ).kernel,
            hf_weight_key=hf_key("per_layer_model_projection.weight"),
            hook_fn=lambda x, _: np.transpose(x),
        )
        loader.port_weight(
            keras_variable=backbone.get_layer(
                "per_layer_projection_norm"
            ).scale,
            hf_weight_key=hf_key("per_layer_projection_norm.weight"),
        )

    vision_encoder = backbone.vision_encoder
    if vision_encoder is not None:
        _convert_vision_encoder(vision_encoder, loader, transformers_config)

    audio_encoder = backbone.audio_encoder
    if audio_encoder is not None:
        _convert_audio_encoder(audio_encoder, loader, transformers_config)

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"decoder_block_{i}")
        _convert_decoder_block(decoder_layer, i, loader, hf_key)

    loader.port_weight(
        keras_variable=backbone.get_layer("final_normalization").scale,
        hf_weight_key=hf_key("norm.weight"),
    )

    return backbone


def _convert_audio_encoder(audio_encoder, loader, transformers_config):
    """Port audio-encoder weights from HF into KerasHub Gemma4AudioEncoder."""
    aud_prefix = "model.audio_tower"
    sscp = audio_encoder.subsample_conv_projection

    for conv_block, hf_attr in [
        (sscp.conv_0, "layer0"),
        (sscp.conv_1, "layer1"),
    ]:
        hf_conv_pfx = f"{aud_prefix}.subsample_conv_projection.{hf_attr}"
        loader.port_weight(
            keras_variable=conv_block.conv.kernel,
            hf_weight_key=f"{hf_conv_pfx}.conv.weight",
            hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
        )
        loader.port_weight(
            keras_variable=conv_block.norm.gamma,
            hf_weight_key=f"{hf_conv_pfx}.norm.weight",
        )

    loader.port_weight(
        keras_variable=sscp.input_proj.kernel,
        hf_weight_key=f"{aud_prefix}.subsample_conv_projection.input_proj_linear.weight",
        hook_fn=lambda x, _: np.transpose(x),
    )

    for i, block in enumerate(audio_encoder.conformer_blocks):
        hf_blk = f"{aud_prefix}.layers.{i}"

        for hf_ffw_name, keras_ffw in [
            ("feed_forward1", block.ffw_start),
            ("feed_forward2", block.ffw_end),
        ]:
            loader.port_weight(
                keras_variable=keras_ffw.ffw_1.kernel,
                hf_weight_key=f"{hf_blk}.{hf_ffw_name}.ffw_layer_1.linear.weight",
                hook_fn=lambda x, _: np.transpose(x),
            )
            loader.port_weight(
                keras_variable=keras_ffw.ffw_2.kernel,
                hf_weight_key=f"{hf_blk}.{hf_ffw_name}.ffw_layer_2.linear.weight",
                hook_fn=lambda x, _: np.transpose(x),
            )

        attn = block.attention.attn
        hf_attn = f"{hf_blk}.self_attn"
        for proj_name, keras_dense in [
            ("q_proj", attn.q_proj),
            ("k_proj", attn.k_proj),
            ("v_proj", attn.v_proj),
        ]:
            loader.port_weight(
                keras_variable=keras_dense.kernel,
                hf_weight_key=f"{hf_attn}.{proj_name}.linear.weight",
                hook_fn=lambda x, _: np.transpose(x),
            )

        loader.port_weight(
            keras_variable=attn.per_dim_scale,
            hf_weight_key=f"{hf_attn}.per_dim_scale",
        )
        loader.port_weight(
            keras_variable=attn.rpe.pos_proj,
            hf_weight_key=f"{hf_attn}.relative_k_proj.weight",
            hook_fn=lambda x, _: np.transpose(x),
        )
        loader.port_weight(
            keras_variable=block.attention.out_proj.kernel,
            hf_weight_key=f"{hf_blk}.self_attn.post.linear.weight",
            hook_fn=lambda x, _: np.transpose(x),
        )

        lconv = block.lconv
        hf_lconv = f"{hf_blk}.lconv1d"
        loader.port_weight(
            keras_variable=lconv.linear_start.kernel,
            hf_weight_key=f"{hf_lconv}.linear_start.linear.weight",
            hook_fn=lambda x, _: np.transpose(x),
        )
        loader.port_weight(
            keras_variable=lconv.depthwise_conv.kernel,
            hf_weight_key=f"{hf_lconv}.depthwise_conv1d.weight",
            hook_fn=lambda x, _: np.transpose(x, (2, 0, 1)),
        )
        loader.port_weight(
            keras_variable=lconv.linear_end.kernel,
            hf_weight_key=f"{hf_lconv}.linear_end.linear.weight",
            hook_fn=lambda x, _: np.transpose(x),
        )

        for hf_ffw_name, keras_ffw in [
            ("feed_forward1", block.ffw_start),
            ("feed_forward2", block.ffw_end),
        ]:
            loader.port_weight(
                keras_variable=keras_ffw.pre_norm.scale,
                hf_weight_key=f"{hf_blk}.{hf_ffw_name}.pre_layer_norm.weight",
            )
            loader.port_weight(
                keras_variable=keras_ffw.post_norm.scale,
                hf_weight_key=f"{hf_blk}.{hf_ffw_name}.post_layer_norm.weight",
            )

        loader.port_weight(
            keras_variable=block.attention.pre_attn_norm.scale,
            hf_weight_key=f"{hf_blk}.norm_pre_attn.weight",
        )
        loader.port_weight(
            keras_variable=block.attention.post_norm.scale,
            hf_weight_key=f"{hf_blk}.norm_post_attn.weight",
        )
        loader.port_weight(
            keras_variable=lconv.pre_norm.scale,
            hf_weight_key=f"{hf_lconv}.pre_layer_norm.weight",
        )
        loader.port_weight(
            keras_variable=lconv.conv_norm.scale,
            hf_weight_key=f"{hf_lconv}.conv_norm.weight",
        )
        loader.port_weight(
            keras_variable=block.norm.scale,
            hf_weight_key=f"{hf_blk}.norm_out.weight",
        )

    # --- Output projection (audio_tower.output_proj) ---
    if audio_encoder.output_proj is not None:
        loader.port_weight(
            keras_variable=audio_encoder.output_proj.kernel,
            hf_weight_key=f"{aud_prefix}.output_proj.weight",
            hook_fn=lambda x, _: np.transpose(x),
        )
        loader.port_weight(
            keras_variable=audio_encoder.output_proj.bias,
            hf_weight_key=f"{aud_prefix}.output_proj.bias",
        )

    # --- Audio output projection (embed_audio.embedding_projection) ---
    loader.port_weight(
        keras_variable=audio_encoder.audio_output_projection.kernel,
        hf_weight_key="model.embed_audio.embedding_projection.weight",
        hook_fn=lambda x, _: np.transpose(x),
    )
    # embed_audio.embedding_post_projection_norm: parameter-free (Gemma4VNorm).


def _convert_vision_encoder(vision_encoder, loader, transformers_config):
    """Port vision-encoder weights from HF into KerasHub vision encoder."""
    image_encoder = vision_encoder.get_layer("image_encoder")
    patch_embedder = image_encoder.patch_embedder

    vis_prefix = "model.vision_tower"

    loader.port_weight(
        keras_variable=patch_embedder.input_proj.kernel,
        hf_weight_key=f"{vis_prefix}.patch_embedder.input_proj.weight",
        hook_fn=lambda x, _: np.transpose(x),
    )
    loader.port_weight(
        keras_variable=patch_embedder.position_embedding_table,
        hf_weight_key=f"{vis_prefix}.patch_embedder.position_embedding_table",
    )

    num_vis_layers = len(image_encoder.encoder_blocks)
    for i in range(num_vis_layers):
        block = image_encoder.encoder_blocks[i]
        vis_layer_prefix = f"{vis_prefix}.encoder.layers.{i}"
        _convert_decoder_block_weights(
            block,
            vis_layer_prefix,
            loader,
        )

    projector_prefix = "model.embed_vision"
    vision_output = vision_encoder.get_layer("vision_output_encoder")
    loader.port_weight(
        keras_variable=vision_output.vision_input_projection.kernel,
        hf_weight_key=f"{projector_prefix}.embedding_projection.weight",
        hook_fn=lambda x, _: np.transpose(x),
    )

    vis_cfg = transformers_config.get("vision_config", {})
    if vis_cfg.get("standardize", False):
        loader.port_weight(
            keras_variable=vision_output.std_bias,
            hf_weight_key=f"{vis_prefix}.std_bias",
        )
        loader.port_weight(
            keras_variable=vision_output.std_scale,
            hf_weight_key=f"{vis_prefix}.std_scale",
        )


def _convert_decoder_block(decoder_layer, layer_idx, loader, hf_key_fn):
    """Port a single text-decoder Gemma4DecoderBlock from HF."""
    layer_prefix = f"layers.{layer_idx}"

    def layer_key(attr):
        return hf_key_fn(f"{layer_prefix}.{attr}")

    if (
        getattr(decoder_layer, "is_kv_shared_layer", False)
        and getattr(decoder_layer, "kv_shared_layer_index", None) is not None
    ):
        kv_layer_prefix = f"layers.{decoder_layer.kv_shared_layer_index}"

        def kv_layer_key(attr):
            return hf_key_fn(f"{kv_layer_prefix}.{attr}")
    else:
        kv_layer_key = layer_key

    # Layer norms
    loader.port_weight(
        keras_variable=decoder_layer.pre_attention_norm.scale,
        hf_weight_key=layer_key("input_layernorm.weight"),
    )
    loader.port_weight(
        keras_variable=decoder_layer.post_attention_norm.scale,
        hf_weight_key=layer_key("post_attention_layernorm.weight"),
    )
    loader.port_weight(
        keras_variable=decoder_layer.pre_ffw_norm.scale,
        hf_weight_key=layer_key("pre_feedforward_layernorm.weight"),
    )
    loader.port_weight(
        keras_variable=decoder_layer.post_ffw_norm.scale,
        hf_weight_key=layer_key("post_feedforward_layernorm.weight"),
    )

    # Attention Q / K / V / O + Q-norm / K-norm
    loader.port_weight(
        keras_variable=decoder_layer.attention.query_dense.kernel,
        hf_weight_key=layer_key("self_attn.q_proj.weight"),
        # HF: [num_q_heads * head_dim, hidden]
        # → Keras: [num_q_heads, hidden, head_dim]
        hook_fn=lambda hf_tensor, keras_shape: np.transpose(
            np.reshape(
                hf_tensor,
                (keras_shape[0], keras_shape[2], keras_shape[1]),
            ),
            axes=(0, 2, 1),
        ),
    )
    loader.port_weight(
        keras_variable=decoder_layer.attention.query_norm.scale,
        hf_weight_key=layer_key("self_attn.q_norm.weight"),
    )
    loader.port_weight(
        keras_variable=decoder_layer.attention.key_dense.kernel,
        hf_weight_key=kv_layer_key("self_attn.k_proj.weight"),
        hook_fn=lambda hf_tensor, keras_shape: np.transpose(
            np.reshape(
                hf_tensor,
                (keras_shape[0], keras_shape[2], keras_shape[1]),
            ),
            axes=(0, 2, 1),
        ),
    )
    loader.port_weight(
        keras_variable=decoder_layer.attention.key_norm.scale,
        hf_weight_key=kv_layer_key("self_attn.k_norm.weight"),
    )
    # v_proj is absent on global-attention layers when attention_k_eq_v=True
    # (26B-A4B, 31B): value reuses the key projection, so value_dense=None.
    if decoder_layer.attention.value_dense is not None:
        loader.port_weight(
            keras_variable=decoder_layer.attention.value_dense.kernel,
            hf_weight_key=kv_layer_key("self_attn.v_proj.weight"),
            hook_fn=lambda hf_tensor, keras_shape: np.transpose(
                np.reshape(
                    hf_tensor,
                    (keras_shape[0], keras_shape[2], keras_shape[1]),
                ),
                axes=(0, 2, 1),
            ),
        )
    # v_norm (Gemma4VNorm) is parameter-free — no weight to port.
    loader.port_weight(
        keras_variable=decoder_layer.attention.output_dense.kernel,
        hf_weight_key=layer_key("self_attn.o_proj.weight"),
        # HF: [hidden, num_q_heads * head_dim]
        # → Keras: [num_q_heads, head_dim, hidden]
        hook_fn=lambda hf_tensor, keras_shape: np.transpose(
            np.reshape(
                hf_tensor,
                (keras_shape[2], keras_shape[0], keras_shape[1]),
            ),
            axes=(1, 2, 0),
        ),
    )

    loader.port_weight(
        keras_variable=decoder_layer.gating_ffw.kernel,
        hf_weight_key=layer_key("mlp.gate_proj.weight"),
        hook_fn=lambda x, _: np.transpose(x),
    )
    loader.port_weight(
        keras_variable=decoder_layer.gating_ffw_2.kernel,
        hf_weight_key=layer_key("mlp.up_proj.weight"),
        hook_fn=lambda x, _: np.transpose(x),
    )
    loader.port_weight(
        keras_variable=decoder_layer.ffw_linear.kernel,
        hf_weight_key=layer_key("mlp.down_proj.weight"),
        hook_fn=lambda x, _: np.transpose(x),
    )

    # Per-layer input conditioning.
    if decoder_layer.hidden_size_per_layer_input > 0:
        loader.port_weight(
            keras_variable=decoder_layer.per_layer_input_gate.kernel,
            hf_weight_key=layer_key("per_layer_input_gate.weight"),
            hook_fn=lambda x, _: np.transpose(x),
        )
        # HF names this `per_layer_projection`; Keras: `per_layer_up_proj`.
        loader.port_weight(
            keras_variable=decoder_layer.per_layer_up_proj.kernel,
            hf_weight_key=layer_key("per_layer_projection.weight"),
            hook_fn=lambda x, _: np.transpose(x),
        )
        loader.port_weight(
            keras_variable=decoder_layer.post_per_layer_input_norm.scale,
            hf_weight_key=layer_key("post_per_layer_input_norm.weight"),
        )

    # MoE block (parallel dense + expert paths).
    if decoder_layer.enable_moe_block:
        # Extra norms.
        loader.port_weight(
            keras_variable=decoder_layer.post_ffw_norm_dense.scale,
            hf_weight_key=layer_key("post_feedforward_layernorm_1.weight"),
        )
        loader.port_weight(
            keras_variable=decoder_layer.pre_ffw_norm_moe.scale,
            hf_weight_key=layer_key("pre_feedforward_layernorm_2.weight"),
        )
        loader.port_weight(
            keras_variable=decoder_layer.post_ffw_norm_moe_path.scale,
            hf_weight_key=layer_key("post_feedforward_layernorm_2.weight"),
        )
        # Router: per-dim scale + projection (rms_norm has no learnable
        # weights).
        loader.port_weight(
            keras_variable=decoder_layer.moe_router.per_dim_scale,
            hf_weight_key=layer_key("router.scale"),
        )
        loader.port_weight(
            keras_variable=decoder_layer.moe_router.proj.kernel,
            hf_weight_key=layer_key("router.proj.weight"),
            hook_fn=lambda x, _: np.transpose(x),
        )
        # Expert bank: HF `gate_up_proj` is [E, 2*I, H], `down_proj` is [E, H,
        # I].
        # Keras Hub `gate` / `up` are [E, H, I], `down` is [E, I, H].
        I = decoder_layer.expert_intermediate_dim
        loader.port_weight(
            keras_variable=decoder_layer.moe_expert_bank.gate_proj,
            hf_weight_key=layer_key("experts.gate_up_proj"),
            hook_fn=lambda x, _: np.transpose(x[:, :I, :], axes=(0, 2, 1)),
        )
        loader.port_weight(
            keras_variable=decoder_layer.moe_expert_bank.up_proj,
            hf_weight_key=layer_key("experts.gate_up_proj"),
            hook_fn=lambda x, _: np.transpose(x[:, I:, :], axes=(0, 2, 1)),
        )
        loader.port_weight(
            keras_variable=decoder_layer.moe_expert_bank.down_proj,
            hf_weight_key=layer_key("experts.down_proj"),
            hook_fn=lambda x, _: np.transpose(x, axes=(0, 2, 1)),
        )
        loader.port_weight(
            keras_variable=decoder_layer.moe_expert_bank.per_expert_scale,
            hf_weight_key=layer_key("router.per_expert_scale"),
        )

    # layer_scalar — present on all text decoder layers (HF Buffer).
    loader.port_weight(
        keras_variable=decoder_layer.layer_scalar,
        hf_weight_key=layer_key("layer_scalar"),
        hook_fn=lambda x, _: np.squeeze(x),
    )


def _convert_decoder_block_weights(block, layer_prefix, loader):
    """Port a vision-encoder Gemma4DecoderBlock using a raw HF prefix string."""

    def key(attr):
        return f"{layer_prefix}.{attr}"

    def port_clips(keras_layer, hf_name):
        if not getattr(keras_layer, "use_clipped_linears", False):
            return
        for w in ["input_min", "input_max", "output_min", "output_max"]:
            loader.port_weight(
                keras_variable=getattr(keras_layer, w),
                hf_weight_key=key(f"{hf_name}.{w}"),
            )

    # Layer norms
    loader.port_weight(
        keras_variable=block.pre_attention_norm.scale,
        hf_weight_key=key("input_layernorm.weight"),
    )
    loader.port_weight(
        keras_variable=block.post_attention_norm.scale,
        hf_weight_key=key("post_attention_layernorm.weight"),
    )
    loader.port_weight(
        keras_variable=block.pre_ffw_norm.scale,
        hf_weight_key=key("pre_feedforward_layernorm.weight"),
    )
    loader.port_weight(
        keras_variable=block.post_ffw_norm.scale,
        hf_weight_key=key("post_feedforward_layernorm.weight"),
    )

    # Attention
    loader.port_weight(
        keras_variable=block.attention.query_dense.dense.kernel,
        hf_weight_key=key("self_attn.q_proj.linear.weight"),
        hook_fn=lambda hf_tensor, keras_shape: np.transpose(
            np.reshape(
                hf_tensor,
                (keras_shape[0], keras_shape[2], keras_shape[1]),
            ),
            axes=(0, 2, 1),
        ),
    )
    loader.port_weight(
        keras_variable=block.attention.query_norm.scale,
        hf_weight_key=key("self_attn.q_norm.weight"),
    )
    loader.port_weight(
        keras_variable=block.attention.key_dense.dense.kernel,
        hf_weight_key=key("self_attn.k_proj.linear.weight"),
        hook_fn=lambda hf_tensor, keras_shape: np.transpose(
            np.reshape(
                hf_tensor,
                (keras_shape[0], keras_shape[2], keras_shape[1]),
            ),
            axes=(0, 2, 1),
        ),
    )
    loader.port_weight(
        keras_variable=block.attention.key_norm.scale,
        hf_weight_key=key("self_attn.k_norm.weight"),
    )
    loader.port_weight(
        keras_variable=block.attention.value_dense.dense.kernel,
        hf_weight_key=key("self_attn.v_proj.linear.weight"),
        hook_fn=lambda hf_tensor, keras_shape: np.transpose(
            np.reshape(
                hf_tensor,
                (keras_shape[0], keras_shape[2], keras_shape[1]),
            ),
            axes=(0, 2, 1),
        ),
    )
    # v_norm: no learnable weight.
    loader.port_weight(
        keras_variable=block.attention.output_dense.dense.kernel,
        hf_weight_key=key("self_attn.o_proj.linear.weight"),
        hook_fn=lambda hf_tensor, keras_shape: np.transpose(
            np.reshape(
                hf_tensor,
                (keras_shape[2], keras_shape[0], keras_shape[1]),
            ),
            axes=(1, 2, 0),
        ),
    )

    # MLP
    loader.port_weight(
        keras_variable=block.gating_ffw.dense.kernel,
        hf_weight_key=key("mlp.gate_proj.linear.weight"),
        hook_fn=lambda x, _: np.transpose(x),
    )
    loader.port_weight(
        keras_variable=block.gating_ffw_2.dense.kernel,
        hf_weight_key=key("mlp.up_proj.linear.weight"),
        hook_fn=lambda x, _: np.transpose(x),
    )
    loader.port_weight(
        keras_variable=block.ffw_linear.dense.kernel,
        hf_weight_key=key("mlp.down_proj.linear.weight"),
        hook_fn=lambda x, _: np.transpose(x),
    )

    # Clipping weights
    port_clips(block.attention.query_dense, "self_attn.q_proj")
    port_clips(block.attention.key_dense, "self_attn.k_proj")
    port_clips(block.attention.value_dense, "self_attn.v_proj")
    port_clips(block.attention.output_dense, "self_attn.o_proj")
    port_clips(block.gating_ffw, "mlp.gate_proj")
    port_clips(block.gating_ffw_2, "mlp.up_proj")
    port_clips(block.ffw_linear, "mlp.down_proj")


def convert_tokenizer(cls, preset, **kwargs):
    tokenizer_config = load_json(preset, "tokenizer.json")
    proto = _build_sentencepiece_proto(tokenizer_config)
    # Check both model.vocab and added_tokens. Vision tokens (e.g. <|image|>)
    # live in model.vocab; audio tokens (e.g. <audio_soft_token>) may only
    # appear in added_tokens for audio-capable models.
    vocab = tokenizer_config.get("model", {}).get("vocab", {})
    added_contents = {
        t["content"] for t in tokenizer_config.get("added_tokens", [])
    }
    has_vision_tokens = (
        IMAGE_PLACEHOLDER_TOKEN in vocab
        or IMAGE_PLACEHOLDER_TOKEN in added_contents
    )
    has_audio_tokens = "<|audio|>" in vocab or "<|audio|>" in added_contents
    return cls(
        proto=proto,
        has_vision_tokens=has_vision_tokens,
        has_audio_tokens=has_audio_tokens,
        **kwargs,
    )


def _build_sentencepiece_proto(tokenizer_config):
    """Build a serialized SentencePiece proto from a tokenizer.json config.

    Gemma4's ``tokenizer.json`` contains a SentencePiece-derived BPE model
    (▁ encoding, byte fallback).  This function reconstructs a valid
    ``ModelProto`` so that ``SentencePieceTokenizer`` can be used for
    inference.
    """
    from sentencepiece import sentencepiece_model_pb2 as sp_pb2

    vocab = dict(tokenizer_config["model"]["vocab"])
    merges = list(tokenizer_config["model"]["merges"])

    # Normalise merge format: [["a","b"], ...] → ["a b", ...].
    if merges and isinstance(merges[0], list) and len(merges[0]) == 2:
        merges = [" ".join(m) for m in merges]

    # Include added / special tokens.
    # All added_tokens entries must tokenize as a single unit (USER_DEFINED
    # in SentencePiece). This mirrors HF fast-tokenizer behaviour where
    # added_tokens are matched before the underlying BPE model runs,
    # regardless of whether their `special` flag is True or False.
    added_token_strings = set()
    special_token_strings = set()  # kept for reference (unused below)
    for token_info in tokenizer_config.get("added_tokens", []):
        content = token_info["content"]
        vocab[content] = token_info["id"]
        added_token_strings.add(content)
        if token_info.get("special", False):
            special_token_strings.add(content)

    # Map each merge result → rank so we can assign piece scores.
    merge_result_rank = {}
    for rank, rule in enumerate(merges):
        parts = rule.split(" ", 1)
        if len(parts) == 2:
            merge_result_rank[parts[0] + parts[1]] = rank

    model_proto = sp_pb2.ModelProto()

    # Trainer spec.
    ts = model_proto.trainer_spec
    ts.model_type = sp_pb2.TrainerSpec.BPE
    ts.vocab_size = len(vocab)
    ts.byte_fallback = True

    # Normalizer spec – replicate the HF normalizer: Replace(" " → "▁").
    ns = model_proto.normalizer_spec
    ns.name = "identity"
    ns.add_dummy_prefix = False
    ns.escape_whitespaces = True
    ns.remove_extra_whitespaces = False

    # Denormalizer spec (for detokenization).
    dns = model_proto.denormalizer_spec
    dns.add_dummy_prefix = False
    dns.escape_whitespaces = True
    dns.remove_extra_whitespaces = False

    # Byte-fallback regex for <0xNN> tokens.
    _byte_re = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")

    # Sentinel score for base characters (lower than any merge).
    base_score = -float(len(merges) + 1)

    # Pieces must be ordered by ID (index == ID in SP protos).
    for token_str, _token_id in sorted(vocab.items(), key=lambda kv: kv[1]):
        piece = model_proto.pieces.add()
        piece.piece = token_str

        if token_str == "<unk>":
            piece.type = sp_pb2.ModelProto.SentencePiece.UNKNOWN
            piece.score = 0.0
        elif _byte_re.match(token_str):
            piece.type = sp_pb2.ModelProto.SentencePiece.BYTE
            piece.score = 0.0
        elif token_str in added_token_strings:
            piece.type = sp_pb2.ModelProto.SentencePiece.USER_DEFINED
            piece.score = 0.0
        elif token_str in merge_result_rank:
            piece.type = sp_pb2.ModelProto.SentencePiece.NORMAL
            piece.score = -float(merge_result_rank[token_str])
        else:
            # Base character – not the result of any merge.
            piece.type = sp_pb2.ModelProto.SentencePiece.NORMAL
            piece.score = base_score

    return model_proto.SerializeToString()


def _resolve_prefix(loader, candidates):
    """Try each candidate prefix in order and return the first that works."""
    for candidate in candidates:
        probe = (
            f"{candidate}.embed_tokens.weight"
            if candidate
            else "embed_tokens.weight"
        )
        try:
            loader.get_tensor(probe)
            return candidate
        except Exception:
            continue
    return candidates[0]
