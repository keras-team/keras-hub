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


def _compute_scale_offset(proc_cfg, section_name):
    """Compute KH scale/offset from HF processor do_rescale/do_normalize flags.

    KH applies: output = input * scale + offset
    HF applies:
        1. Rescale (if do_rescale): x = x * rescale_factor
        2. Normalize (if do_normalize): x = (x - mean) / std
    Combined:   scale = rescale_factor / std,  offset = -mean / std

    If neither flag is set the transform is identity (scale=None, offset=None).
    If do_resize=False an error is raised because Gemma4ImageConverter always
    resizes; this configuration is not supported.
    """
    do_rescale = proc_cfg["do_rescale"]
    do_normalize = proc_cfg["do_normalize"]
    do_resize = proc_cfg["do_resize"]

    if not do_resize:
        raise ValueError(
            f"{section_name}: do_resize=False is not supported by "
            "Gemma4ImageConverter, which always applies aspect-ratio-"
            "preserving resize."
        )

    if not do_rescale and not do_normalize:
        return None, None

    effective_rescale = proc_cfg["rescale_factor"] if do_rescale else 1.0
    effective_mean = proc_cfg["image_mean"] if do_normalize else [0.0, 0.0, 0.0]
    effective_std = proc_cfg["image_std"] if do_normalize else [1.0, 1.0, 1.0]

    scale = [(effective_rescale / s) for s in effective_std]
    offset = [(-m / s) for m, s in zip(effective_mean, effective_std)]
    return scale, offset


def load_image_converter_config(preset, transformers_config):
    """Return kwargs for Gemma4ImageConverter, or None for text-only models."""
    if "vision_config" not in transformers_config:
        return None

    processor_config = load_json(preset, "processor_config.json")
    image_processor = processor_config["image_processor"]
    scale, offset = _compute_scale_offset(image_processor, "image_processor")

    vision_config = transformers_config["vision_config"]
    # image_size is not present in the HF vision_config; 896 is the fixed
    # positional-embedding size used across all Gemma4 vision checkpoints
    # (matches the hardcoded value in convert_backbone_config).
    image_size = 896
    patch_size = vision_config["patch_size"]
    # max_soft_tokens lives in the image_processor section of processor_config;
    # vision_soft_tokens_per_image in the model config is the authoritative cap.
    max_soft_tokens = image_processor["max_soft_tokens"]
    pooling_kernel_size = vision_config["pooling_kernel_size"]

    return {
        "image_size": (image_size, image_size),
        "patch_size": patch_size,
        "max_soft_tokens": max_soft_tokens,
        "pooling_kernel_size": pooling_kernel_size,
        "scale": scale,
        "offset": offset,
    }


def load_audio_converter_config(preset, transformers_config):
    """Return Gemma4AudioConverter kwargs, or None for text/vision models."""
    if not transformers_config.get("audio_config"):
        return None

    processor_config = load_json(preset, "processor_config.json")
    feature_extractor = processor_config["feature_extractor"]

    # Map HF keys to KerasHub keys
    return {
        "num_mels": feature_extractor["feature_size"],
        "num_fft_bins": feature_extractor["fft_length"],
        "stride": feature_extractor["hop_length"],
        "sampling_rate": feature_extractor["sampling_rate"],
        "frame_length": feature_extractor["frame_length"],
        "max_frequency": feature_extractor["max_frequency"],
        "min_frequency": feature_extractor["min_frequency"],
        "mel_floor": feature_extractor["mel_floor"],
    }


def load_video_converter_config(preset, transformers_config):
    """Return Gemma4VideoConverter kwargs, or None if not a video model."""
    processor_config = load_json(preset, "processor_config.json")
    if "video_processor" not in processor_config:
        return None

    video_proc = processor_config["video_processor"]
    scale, offset = _compute_scale_offset(video_proc, "video_processor")

    return {
        "num_frames": video_proc["num_frames"],
        "max_soft_tokens": video_proc["max_soft_tokens"],
        "patch_size": video_proc["patch_size"],
        "pooling_kernel_size": video_proc["pooling_kernel_size"],
        "scale": scale,
        "offset": offset,
    }


def load_preprocessor_config(preset, transformers_config):
    """Return extra Gemma4CausalLMPreprocessor kwargs from processor_config."""
    processor_config = load_json(preset, "processor_config.json")
    if "video_processor" not in processor_config:
        return {}

    video_proc = processor_config["video_processor"]
    # do_sample_frames=True means the processor samples num_frames from the
    # full video. KH's VideoConverter always samples; if this flag is ever
    # False, frames are expected to be pre-sampled by the caller and num_frames
    # is ignored at runtime (the converter still linspaces over whatever it
    # receives, but total_frames == num_frames so the result is identical).
    return {
        "num_frames_per_video": video_proc["num_frames"],
        "num_vision_tokens_per_frame": video_proc["max_soft_tokens"],
        # video_fps is not stored in HF configs; 24.0 matches the HF default.
        "video_fps": 24.0,
    }


def convert_backbone_config(transformers_config):
    """Map a Transformers config dict → Gemma4Backbone keyword arguments."""
    model_type = transformers_config.get("model_type", "gemma4")
    is_text_only = model_type == "gemma4_text"

    if is_text_only:
        text_cfg = transformers_config
        vision_encoder = None
        audio_encoder = None
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
            audio_encoder = Gemma4AudioEncoder(
                hidden_size=aud_cfg["hidden_size"],
                num_heads=aud_cfg["num_attention_heads"],
                num_layers=aud_cfg["num_hidden_layers"],
                chunk_size=aud_cfg["attention_chunk_size"],
                context_left=aud_cfg["attention_context_left"],
                context_right=aud_cfg["attention_context_right"],
                logit_cap=aud_cfg["attention_logit_cap"],
                invalid_logit_value=aud_cfg["attention_invalid_logits_value"],
                conv_kernel_size=aud_cfg["conv_kernel_size"],
                residual_weight=aud_cfg["residual_weight"],
                gradient_clipping=aud_cfg["gradient_clipping"],
                sscp_conv_channels=tuple(aud_cfg["subsampling_conv_channels"]),
                output_proj_dims=aud_cfg["output_proj_dims"],
                output_dim=text_cfg["hidden_size"],
                norm_eps=aud_cfg["rms_norm_eps"],
                sscp_norm_eps=aud_cfg["rms_norm_eps"],
            )
        else:
            audio_encoder = None

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
    print(f"DEBUG convert_gemma4: backbone.hidden_size_per_layer_input = {backbone.hidden_size_per_layer_input}")
    if backbone.hidden_size_per_layer_input > 0:
        print("DEBUG convert_gemma4: Porting per-layer token embedding")
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

    def port_clips(keras_layer, hf_name):
        if not getattr(keras_layer, "use_clipped_linears", False):
            return
        for w in ["input_min", "input_max", "output_min", "output_max"]:
            loader.port_weight(
                keras_variable=getattr(keras_layer, w),
                hf_weight_key=f"{hf_name}.{w}",
            )


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
                keras_variable=keras_ffw.ffw_1.dense.kernel,
                hf_weight_key=f"{hf_blk}.{hf_ffw_name}.ffw_layer_1.linear.weight",
                hook_fn=lambda x, _: np.transpose(x),
            )

            loader.port_weight(
                keras_variable=keras_ffw.ffw_2.dense.kernel,
                hf_weight_key=f"{hf_blk}.{hf_ffw_name}.ffw_layer_2.linear.weight",
                hook_fn=lambda x, _: np.transpose(x),
            )

            port_clips(keras_ffw.ffw_1, f"{hf_blk}.{hf_ffw_name}.ffw_layer_1")
            port_clips(keras_ffw.ffw_2, f"{hf_blk}.{hf_ffw_name}.ffw_layer_2")

        attn = block.attention.attn
        hf_attn = f"{hf_blk}.self_attn"
        for proj_name, keras_dense in [
            ("q_proj", attn.q_proj),
            ("k_proj", attn.k_proj),
            ("v_proj", attn.v_proj),
        ]:
            loader.port_weight(
                keras_variable=keras_dense.dense.kernel,
                hf_weight_key=f"{hf_attn}.{proj_name}.linear.weight",
                hook_fn=lambda x, _: np.transpose(x),
            )
            port_clips(keras_dense, f"{hf_attn}.{proj_name}")



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
            keras_variable=block.attention.out_proj.dense.kernel,
            hf_weight_key=f"{hf_blk}.self_attn.post.linear.weight",
            hook_fn=lambda x, _: np.transpose(x),
        )
        port_clips(block.attention.out_proj, f"{hf_blk}.self_attn.post")



        lconv = block.lconv
        hf_lconv = f"{hf_blk}.lconv1d"
        loader.port_weight(
            keras_variable=lconv.linear_start.dense.kernel,
            hf_weight_key=f"{hf_lconv}.linear_start.linear.weight",
            hook_fn=lambda x, _: np.transpose(x),
        )
        port_clips(lconv.linear_start, f"{hf_lconv}.linear_start")


        loader.port_weight(
            keras_variable=lconv.depthwise_conv.kernel,
            hf_weight_key=f"{hf_lconv}.depthwise_conv1d.weight",
            hook_fn=lambda x, _: np.transpose(x, (2, 0, 1)),
        )
        loader.port_weight(
            keras_variable=lconv.linear_end.dense.kernel,
            hf_weight_key=f"{hf_lconv}.linear_end.linear.weight",
            hook_fn=lambda x, _: np.transpose(x),
        )
        port_clips(lconv.linear_end, f"{hf_lconv}.linear_end")



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
    # KV-shared layers have no k/v weights of their own (HF PR #45336).
    if not decoder_layer.attention.is_kv_shared_layer:
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
    has_audio_tokens = (
        "<|audio|>" in vocab
        or "<|audio|>" in added_contents
    )
    return cls(
        proto=proto,
        has_vision_tokens=has_vision_tokens,
        has_audio_tokens=has_audio_tokens,
        has_video_tokens=True,
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

    # Manually add video token if missing (needed for Gemma4 video models)
    if "<|video|>" not in vocab and "<|video|>" not in added_token_strings:
        vocab["<|video|>"] = 258884
        added_token_strings.add("<|video|>")

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
